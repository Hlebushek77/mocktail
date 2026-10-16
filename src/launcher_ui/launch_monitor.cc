// mocktail_launch_monitor: tiny detached window shown by the launcher on
// Play. Watches the session-log directory for the newest log, shows launch
// phases as a progress bar and closes itself when the game is up.
// Usage: mocktail_launch_monitor <sessions-dir>
#include <adwaita.h>
#include <gio/gio.h>
#include <glib/gstdio.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Phase {
  const char* marker;
  int percent;
  const char* label;
};

// Ordered by appearance in a healthy session log.
constexpr Phase kPhases[] = {
    {"[mocktail] version=", 5, "Starting..."},
    {"Bionic linker", 20, "Linker..."},
    {"Library path:", 30, "Loading Roblox..."},
    {"Registering Android", 40, "Android runtime..."},
    {"StartLuaAppDM", 55, "Starting LuaApp..."},
    {"setStage:", 70, "Initializing..."},
    {"DataModel Loading", 80, "Loading game..."},
    {"setStage: (stage:LuaApp)", 92, "Almost there..."},
};

constexpr const char* kReadyMarkers[] = {
    "first Roblox frame presented",
    "SwapBuffers",
};

GtkTextBuffer* g_log_buffer = nullptr;
GtkWidget* g_progress = nullptr;
GtkWidget* g_status = nullptr;
GtkWidget* g_window = nullptr;
int g_phase = 0;
bool g_done = false;
std::string g_tailed_path;

void AppendLogLine(const std::string& line) {
  if (line.find("Gtk-WARNING") != std::string::npos) return;
  GtkTextIter end;
  gtk_text_buffer_get_end_iter(g_log_buffer, &end);
  gtk_text_buffer_insert(g_log_buffer, &end, line.c_str(), -1);
  gtk_text_buffer_insert(g_log_buffer, &end, "\n", -1);
}

void SetPhase(int percent, const std::string& label) {
  if (percent < g_phase) return;
  g_phase = percent;
  gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(g_progress),
                                percent / 100.0);
  gtk_label_set_text(GTK_LABEL(g_status), label.c_str());
}

void CheckDone() {
  if (g_done) return;
  g_done = true;
  gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(g_progress), 1.0);
  gtk_label_set_text(GTK_LABEL(g_status), "Game is running. Have fun!");
  g_timeout_add_seconds(
      3,
      [](gpointer data) -> gboolean {
        gtk_window_close(GTK_WINDOW(data));
        return G_SOURCE_REMOVE;
      },
      g_window);
}

void OnLine(const std::string& line) {
  AppendLogLine(line);
  for (const Phase& phase : kPhases) {
    if (line.find(phase.marker) != std::string::npos) {
      SetPhase(phase.percent, phase.label);
    }
  }
  for (const char* marker : kReadyMarkers) {
    if (line.find(marker) != std::string::npos) {
      CheckDone();
      return;
    }
  }
  if (line.find("[FATAL]") != std::string::npos ||
      line.find("Segmentation fault") != std::string::npos) {
    gtk_label_set_text(
        GTK_LABEL(g_status),
        "Failed - see the log above. The window stays open.");
  }
}

struct TailState {
  std::string path;
  GFileMonitor* monitor = nullptr;
  guint64 offset = 0;
  std::string pending;
};

void ReadAvailable(TailState* state, GFile* file) {
  GError* error = nullptr;
  GFileInputStream* stream = g_file_read(file, nullptr, &error);
  if (stream == nullptr) {
    if (error != nullptr) g_error_free(error);
    return;
  }
  if (!g_seekable_seek(G_SEEKABLE(stream), state->offset, G_SEEK_SET, nullptr,
                       nullptr)) {
    g_object_unref(stream);
    return;
  }
  char buffer[65536];
  while (true) {
    gssize count = g_input_stream_read(G_INPUT_STREAM(stream), buffer,
                                       sizeof(buffer), nullptr, nullptr);
    if (count <= 0) break;
    state->offset += static_cast<guint64>(count);
    state->pending.append(buffer, static_cast<size_t>(count));
    size_t pos = 0;
    while (true) {
      size_t end = state->pending.find('\n', pos);
      if (end == std::string::npos) break;
      OnLine(state->pending.substr(pos, end - pos));
      pos = end + 1;
    }
    state->pending.erase(0, pos);
  }
  g_object_unref(stream);
}

// Newest *.log in the sessions dir; empty if none yet.
std::string NewestLog(const std::string& dir) {
  GDir* listing = g_dir_open(dir.c_str(), 0, nullptr);
  if (listing == nullptr) return {};
  std::string best;
  guint64 best_mtime = 0;
  const char* name = nullptr;
  while ((name = g_dir_read_name(listing)) != nullptr) {
    size_t len = std::strlen(name);
    if (len < 5 || std::strcmp(name + len - 4, ".log") != 0) continue;
    std::string full = dir + "/" + name;
    GStatBuf info;
    if (g_stat(full.c_str(), &info) != 0) continue;
    if (static_cast<guint64>(info.st_mtime) >= best_mtime) {
      best_mtime = static_cast<guint64>(info.st_mtime);
      best = full;
    }
  }
  g_dir_close(listing);
  return best;
}

struct MonitorState {
  std::string dir;
  TailState tail;
  GFileMonitor* dir_monitor = nullptr;
};

void PollSessions(gpointer data);

void OnDirChanged(GFileMonitor*, GFile*, GFile*, GFileMonitorEvent,
                  gpointer data) {
  PollSessions(data);
}

void PollSessions(gpointer data) {
  auto* state = static_cast<MonitorState*>(data);
  const std::string newest = NewestLog(state->dir);
  if (newest.empty() || newest == state->tail.path) return;
  // A new session log appeared (the game started writing): follow it.
  state->tail.path = newest;
  state->tail.offset = 0;
  state->tail.pending.clear();
  GFile* file = g_file_new_for_path(newest.c_str());
  ReadAvailable(&state->tail, file);
  g_object_unref(file);
}

void Activate(GApplication* application, gpointer data) {
  const char* sessions_dir = static_cast<const char*>(data);
  g_window = adw_application_window_new(GTK_APPLICATION(application));
  gtk_window_set_title(GTK_WINDOW(g_window), "Запуск");
  gtk_window_set_default_size(GTK_WINDOW(g_window), 620, 420);
  gtk_window_set_icon_name(GTK_WINDOW(g_window), "space.bigrat.mocktail");

  GtkWidget* box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
  gtk_widget_set_margin_top(box, 12);
  gtk_widget_set_margin_bottom(box, 12);
  gtk_widget_set_margin_start(box, 12);
  gtk_widget_set_margin_end(box, 12);

  GtkWidget* head = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
  GtkWidget* icon = gtk_image_new_from_icon_name("space.bigrat.mocktail");
  gtk_image_set_pixel_size(GTK_IMAGE(icon), 40);
  gtk_box_append(GTK_BOX(head), icon);
  GtkWidget* title = gtk_label_new("Запуск Roblox…");
  gtk_label_set_xalign(GTK_LABEL(title), 0.0);
  gtk_widget_add_css_class(title, "title-2");
  gtk_box_append(GTK_BOX(head), title);
  gtk_box_append(GTK_BOX(box), head);

  GtkWidget* progress = gtk_progress_bar_new();
  g_progress = progress;
  gtk_progress_bar_set_show_text(GTK_PROGRESS_BAR(progress), TRUE);
  gtk_box_append(GTK_BOX(box), progress);

  g_status = gtk_label_new("Waiting for the game log…");
  gtk_label_set_xalign(GTK_LABEL(g_status), 0.0);
  gtk_box_append(GTK_BOX(box), g_status);

  GtkWidget* scrolled = gtk_scrolled_window_new();
  gtk_widget_set_vexpand(scrolled, TRUE);
  GtkWidget* view = gtk_text_view_new();
  gtk_text_view_set_editable(GTK_TEXT_VIEW(view), FALSE);
  gtk_text_view_set_cursor_visible(GTK_TEXT_VIEW(view), FALSE);
  gtk_text_view_set_monospace(GTK_TEXT_VIEW(view), TRUE);
  g_log_buffer = gtk_text_view_get_buffer(GTK_TEXT_VIEW(view));
  g_signal_connect_swapped(
      g_log_buffer, "changed",
      G_CALLBACK(+[](GtkTextView* v) {
        GtkTextIter end;
        GtkTextBuffer* b = gtk_text_view_get_buffer(v);
        gtk_text_buffer_get_end_iter(b, &end);
        gtk_text_view_scroll_to_iter(v, &end, 0.0, FALSE, 0.0, 0.0);
      }),
      view);
  gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(scrolled), view);
  GtkWidget* frame = gtk_frame_new("Logs:");
  gtk_frame_set_child(GTK_FRAME(frame), scrolled);
  gtk_box_append(GTK_BOX(box), frame);

  adw_application_window_set_content(ADW_APPLICATION_WINDOW(g_window), box);
  gtk_window_present(GTK_WINDOW(g_window));

  auto* state = new MonitorState();
  state->dir = sessions_dir;
  GFile* dir_file = g_file_new_for_path(sessions_dir);
  state->dir_monitor = g_file_monitor_directory(
      dir_file, G_FILE_MONITOR_NONE, nullptr, nullptr);
  if (state->dir_monitor != nullptr) {
    g_signal_connect(state->dir_monitor, "changed",
                     G_CALLBACK(OnDirChanged), state);
  }
  g_object_unref(dir_file);
  PollSessions(state);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fputs("usage: mocktail_launch_monitor <sessions-dir>\n", stderr);
    return 2;
  }
  AdwApplication* application = adw_application_new(
      "space.bigrat.mocktail.LaunchMonitor", G_APPLICATION_NON_UNIQUE);
  g_signal_connect(application, "activate", G_CALLBACK(Activate), argv[1]);
  int status = g_application_run(G_APPLICATION(application), 0, nullptr);
  g_object_unref(application);
  return status;
}
