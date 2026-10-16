// Extras page ("Дополнительно"): launcher-local customization that never
// touches config.yaml (the game rejects unknown keys). Backed by
// launcher.json via ui_prefs.h. Theme and accent apply live; language and
// tab layout apply on restart.
//
// NOTE: no braced lists or multi-comma expressions inside G_CALLBACK(...:
// the preprocessor splits macro arguments on top-level commas, and braces
// do not protect them. All handlers are plain static functions.
#include <adwaita.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "launcher_ui/bindings.h"
#include "launcher_ui/i18n.h"
#include "launcher_ui/launcher_context.h"
#include "launcher_ui/pages.h"
#include "launcher_ui/ui_prefs.h"

namespace mocktail::launcher_ui {
namespace {

constexpr const char* kThemeValues[] = {"system", "dark", "light"};
constexpr const char* kAccentValues[] = {"default", "blue", "green",
                                         "orange", "purple", "red"};
constexpr const char* kLanguageValues[] = {"", "en", "ru"};

GtkWidget* MakeCombo(const std::vector<std::string>& labels, int selected) {
  GtkStringList* model = gtk_string_list_new(nullptr);
  for (const std::string& label : labels) {
    gtk_string_list_append(model, label.c_str());
  }
  GtkWidget* combo = adw_combo_row_new();
  adw_combo_row_set_model(ADW_COMBO_ROW(combo), G_LIST_MODEL(model));
  adw_combo_row_set_selected(ADW_COMBO_ROW(combo), selected);
  return combo;
}

int IndexOf(const std::string& value, const char* const* values,
            std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    if (value == values[i]) return static_cast<int>(i);
  }
  return 0;
}

void OnThemeSelected(GObject* object, GParamSpec*, gpointer) {
  const guint selected = adw_combo_row_get_selected(ADW_COMBO_ROW(object));
  if (selected > 2) return;
  UiPrefs prefs = LoadUiPrefs();
  prefs.theme = kThemeValues[selected];
  if (SaveUiPrefs(prefs)) ApplyUiTheme();
}

void OnAccentSelected(GObject* object, GParamSpec*, gpointer) {
  const guint selected = adw_combo_row_get_selected(ADW_COMBO_ROW(object));
  if (selected > 5) return;
  UiPrefs prefs = LoadUiPrefs();
  prefs.accent = kAccentValues[selected];
  if (SaveUiPrefs(prefs)) ApplyUiAccent();
}

void OnLanguageSelected(GObject* object, GParamSpec*, gpointer data) {
  const guint selected = adw_combo_row_get_selected(ADW_COMBO_ROW(object));
  if (selected > 2) return;
  UiPrefs prefs = LoadUiPrefs();
  prefs.language = kLanguageValues[selected];
  if (!SaveUiPrefs(prefs)) return;
  auto* context = static_cast<LauncherContext*>(data);
  if (context != nullptr) {
    context->Toast(_("Restart the launcher to apply the language"));
  }
}

GtkWidget* BuildThemeRow(LauncherContext* context) {
  UiPrefs prefs = LoadUiPrefs();
  std::vector<std::string> labels;
  labels.push_back(_("System"));
  labels.push_back(_("Dark"));
  labels.push_back(_("Light"));
  GtkWidget* row = MakeCombo(labels, IndexOf(prefs.theme, kThemeValues, 3));
  RowSpec spec;
  spec.title = _("Theme");
  spec.keywords = {"theme", "dark", "light", "тема", "тёмная", "светлая",
                   "оформление"};
  spec.hint.subtitle = _("Applies immediately");
  g_signal_connect(row, "notify::selected", G_CALLBACK(OnThemeSelected),
                   nullptr);
  return DecorateRow(context, row, std::move(spec));
}

GtkWidget* BuildAccentRow(LauncherContext* context) {
  UiPrefs prefs = LoadUiPrefs();
  std::vector<std::string> labels;
  labels.push_back(_("Default"));
  labels.push_back(_("Blue"));
  labels.push_back(_("Green"));
  labels.push_back(_("Orange"));
  labels.push_back(_("Purple"));
  labels.push_back(_("Red"));
  GtkWidget* row = MakeCombo(labels, IndexOf(prefs.accent, kAccentValues, 6));
  RowSpec spec;
  spec.title = _("Accent color");
  spec.keywords = {"accent", "color", "акцент", "цвет"};
  spec.hint.subtitle = _("Applies immediately");
  g_signal_connect(row, "notify::selected", G_CALLBACK(OnAccentSelected),
                   nullptr);
  return DecorateRow(context, row, std::move(spec));
}

GtkWidget* BuildLanguageRow(LauncherContext* context) {
  UiPrefs prefs = LoadUiPrefs();
  std::vector<std::string> labels;
  labels.push_back(_("System language"));
  labels.push_back(_("English"));
  labels.push_back("Русский");
  GtkWidget* row =
      MakeCombo(labels, IndexOf(prefs.language, kLanguageValues, 3));
  RowSpec spec;
  spec.title = _("Interface language");
  spec.keywords = {"language", "locale", "язык", "интерфейс", "перевод"};
  spec.hint.subtitle = _("Applies on restart");
  g_signal_connect(row, "notify::selected", G_CALLBACK(OnLanguageSelected),
                   context);
  return DecorateRow(context, row, std::move(spec));
}

struct TabCtl {
  LauncherContext* context;
  std::size_t row;
};

struct HideCtl {
  LauncherContext* context;
  std::string id;
};

void OnTabUp(GtkButton*, gpointer data) {
  std::unique_ptr<TabCtl> ctl(static_cast<TabCtl*>(data));
  if (ctl->row == 0) return;
  UiPrefs prefs = LoadUiPrefs();
  std::vector<std::string> ids;
  for (const char* id : OrderedTabIds()) ids.emplace_back(id);
  if (ctl->row < ids.size()) {
    std::swap(ids[ctl->row - 1], ids[ctl->row]);
    prefs.tab_order = ids;
    SaveUiPrefs(prefs);
    ctl->context->Toast(_("Tab order saved, applies on restart"));
  }
}

void OnTabDown(GtkButton*, gpointer data) {
  std::unique_ptr<TabCtl> ctl(static_cast<TabCtl*>(data));
  UiPrefs prefs = LoadUiPrefs();
  std::vector<std::string> ids;
  for (const char* id : OrderedTabIds()) ids.emplace_back(id);
  if (ctl->row + 1 < ids.size()) {
    std::swap(ids[ctl->row], ids[ctl->row + 1]);
    prefs.tab_order = ids;
    SaveUiPrefs(prefs);
    ctl->context->Toast(_("Tab order saved, applies on restart"));
  }
}

void OnTabToggled(GObject* object, GParamSpec*, gpointer data) {
  std::unique_ptr<HideCtl> ctl(static_cast<HideCtl*>(data));
  UiPrefs prefs = LoadUiPrefs();
  const bool active = gtk_switch_get_active(GTK_SWITCH(object)) != FALSE;
  auto& hidden = prefs.hidden_tabs;
  hidden.erase(std::remove(hidden.begin(), hidden.end(), ctl->id),
               hidden.end());
  if (!active) hidden.push_back(ctl->id);
  SaveUiPrefs(prefs);
}

// One row per sidebar section: visibility switch + up/down order buttons.
// Applies on restart (sidebar and pages are built once at startup).
GtkWidget* BuildTabsGroup(GtkWidget* page, LauncherContext* context) {
  GtkWidget* group =
      AddGroup(page, _("Sidebar tabs"),
               _("Show, hide and reorder sections. Applies on restart."));
  UiPrefs prefs = LoadUiPrefs();
  auto is_hidden = [&](const char* id) {
    for (const std::string& h : prefs.hidden_tabs) {
      if (h == id) return true;
    }
    return false;
  };
  std::vector<Section> ordered;
  for (const char* id : OrderedTabIds()) {
    for (const SectionInfo& info : Sections()) {
      if (std::string(info.id) == id) {
        ordered.push_back(info.section);
        break;
      }
    }
  }
  for (std::size_t row = 0; row < ordered.size(); ++row) {
    const SectionInfo& info = GetSectionInfo(ordered[row]);
    GtkWidget* line = adw_action_row_new();
    adw_preferences_row_set_title(ADW_PREFERENCES_ROW(line), _(info.title));
    GtkWidget* box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);
    GtkWidget* up = gtk_button_new_from_icon_name("go-up-symbolic");
    GtkWidget* down = gtk_button_new_from_icon_name("go-down-symbolic");
    gtk_widget_set_tooltip_text(up, _("Move up"));
    gtk_widget_set_tooltip_text(down, _("Move down"));
    GtkWidget* toggle = gtk_switch_new();
    gtk_switch_set_active(GTK_SWITCH(toggle),
                          !is_hidden(info.id) ? TRUE : FALSE);
    gtk_widget_set_tooltip_text(toggle, _("Show in sidebar"));
    gtk_widget_set_sensitive(toggle, std::string(info.id) != "extras");
    gtk_box_append(GTK_BOX(box), up);
    gtk_box_append(GTK_BOX(box), down);
    gtk_box_append(GTK_BOX(box), toggle);
    adw_action_row_add_suffix(ADW_ACTION_ROW(line), box);
    RowSpec spec;
    spec.keywords = {"tabs", "sidebar", "вкладки", "порядок", "скрыть"};
    auto* up_ctl = new TabCtl{context, row};
    auto* down_ctl = new TabCtl{context, row};
    auto* hide_ctl = new HideCtl{context, info.id};
    g_signal_connect(up, "clicked", G_CALLBACK(OnTabUp), up_ctl);
    g_signal_connect(down, "clicked", G_CALLBACK(OnTabDown), down_ctl);
    g_signal_connect(toggle, "notify::active", G_CALLBACK(OnTabToggled),
                     hide_ctl);
    AddRow(group, DecorateRow(context, line, std::move(spec)));
  }
  return group;
}

}  // namespace

// Extras (P1 part 2): launcher-local customization.
GtkWidget* BuildExtrasPage(LauncherContext* context) {
  GtkWidget* page =
      NewPage(context, Section::kExtras, _("Look, language and sidebar"));
  GtkWidget* look =
      AddGroup(page, _("Look"), _("Theme and accent, applied immediately"));
  AddRow(look, BuildThemeRow(context));
  AddRow(look, BuildAccentRow(context));
  GtkWidget* lang = AddGroup(page, _("Language"), _("Applies on restart"));
  AddRow(lang, BuildLanguageRow(context));
  BuildTabsGroup(page, context);
  return page;
}

}  // namespace mocktail::launcher_ui
