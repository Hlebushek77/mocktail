// Changelog page: maintainer-curated release notes (title + text +
// optional photo per entry), rendered from changelog.json. The file lives
// next to the other metadata (config/changelog.json in the repo, installed
// under <datadir>/mocktail/metadata); MOCKTAIL_CHANGELOG_PATH overrides it.
#include <adwaita.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

#include <nlohmann/json.hpp>

#include "launcher_ui/bindings.h"
#include "launcher_ui/i18n.h"
#include "launcher_ui/launcher_context.h"
#include "launcher_ui/pages.h"

#ifndef MOCKTAIL_CHANGELOG_PATH
#define MOCKTAIL_CHANGELOG_PATH "config/changelog.json"
#endif

namespace mocktail::launcher_ui {
namespace {

struct ChangelogEntry {
  std::string version;
  std::string date;
  std::string title;
  std::string text;
  std::string image;
};

std::filesystem::path ExeDir() {
  char buf[4096] = {};
  ssize_t len = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (len <= 0) return {};
  return std::filesystem::path(std::string(buf, static_cast<size_t>(len)))
      .parent_path();
}

std::filesystem::path FindChangelog() {
  if (const char* override_path = std::getenv("MOCKTAIL_CHANGELOG_PATH")) {
    std::error_code error;
    if (std::filesystem::is_regular_file(override_path, error)) {
      return override_path;
    }
  }
  const std::filesystem::path exe = ExeDir();
  if (!exe.empty()) {
    const std::filesystem::path installed =
        exe / "../share/mocktail/metadata/changelog.json";
    std::error_code error;
    if (std::filesystem::is_regular_file(installed, error)) {
      return installed.lexically_normal();
    }
  }
  {
    std::error_code error;
    if (std::filesystem::is_regular_file(MOCKTAIL_CHANGELOG_PATH, error)) {
      return MOCKTAIL_CHANGELOG_PATH;
    }
  }
  return {};
}

std::string StrField(const nlohmann::json& object, const char* key) {
  auto found = object.find(key);
  if (found == object.end() || !found->is_string()) return {};
  return found->get<std::string>();
}

std::vector<ChangelogEntry> LoadChangelog(std::string* error) {
  std::vector<ChangelogEntry> entries;
  const std::filesystem::path path = FindChangelog();
  if (path.empty()) {
    if (error != nullptr) *error = _("changelog.json was not found");
    return entries;
  }
  std::ifstream file(path);
  if (!file) {
    if (error != nullptr) *error = _("changelog.json cannot be read");
    return entries;
  }
  nlohmann::json document;
  try {
    document = nlohmann::json::parse(file);
  } catch (const std::exception& e) {
    if (error != nullptr) {
      *error = Format(_("changelog.json is not valid JSON: %s"), e.what());
    }
    return entries;
  }
  if (!document.is_array()) {
    if (error != nullptr) *error = _("changelog.json must hold an array");
    return entries;
  }
  const std::filesystem::path base = path.parent_path();
  for (const nlohmann::json& item : document) {
    if (!item.is_object()) continue;
    ChangelogEntry entry;
    entry.version = StrField(item, "version");
    entry.date = StrField(item, "date");
    entry.title = StrField(item, "title");
    entry.text = StrField(item, "text");
    entry.image = StrField(item, "image");
    if (entry.title.empty() && entry.text.empty()) continue;
    if (!entry.image.empty()) {
      std::filesystem::path image_path(entry.image);
      if (image_path.is_relative()) image_path = base / image_path;
      std::error_code image_error;
      if (std::filesystem::is_regular_file(image_path, image_error)) {
        entry.image = image_path.string();
      } else {
        entry.image.clear();
      }
    }
    entries.push_back(std::move(entry));
    if (entries.size() >= 50) break;
  }
  return entries;
}

GtkWidget* BuildEntryGroup(GtkWidget* page, const ChangelogEntry& entry) {
  std::string heading = entry.version;
  if (!entry.date.empty()) {
    if (!heading.empty()) heading += " · ";
    heading += entry.date;
  }
  if (heading.empty()) heading = _("Release notes");
  GtkWidget* group =
      AddGroup(page, heading,
               entry.title.empty() ? _("Release notes") : entry.title);
  if (!entry.text.empty()) {
    GtkWidget* label = gtk_label_new(entry.text.c_str());
    gtk_label_set_wrap(GTK_LABEL(label), TRUE);
    gtk_label_set_xalign(GTK_LABEL(label), 0.0);
    gtk_label_set_selectable(GTK_LABEL(label), TRUE);
    gtk_widget_set_margin_top(label, 6);
    gtk_widget_set_margin_bottom(label, 6);
    gtk_widget_set_margin_start(label, 12);
    gtk_widget_set_margin_end(label, 12);
    adw_preferences_group_add(ADW_PREFERENCES_GROUP(group), label);
  }
  if (!entry.image.empty()) {
    GtkWidget* picture = gtk_picture_new_for_filename(entry.image.c_str());
    if (picture != nullptr) {
      gtk_picture_set_content_fit(GTK_PICTURE(picture),
                                  GTK_CONTENT_FIT_SCALE_DOWN);
      gtk_widget_set_size_request(picture, 480, 270);
      gtk_widget_set_margin_bottom(picture, 6);
      gtk_widget_set_margin_start(picture, 12);
      gtk_widget_set_margin_end(picture, 12);
      adw_preferences_group_add(ADW_PREFERENCES_GROUP(group), picture);
    }
  }
  return group;
}

}  // namespace

// Changelog (P1): one group per release entry, newest first as stored.
GtkWidget* BuildChangelogPage(LauncherContext* context) {
  GtkWidget* page =
      NewPage(context, Section::kChangelog, _("What changed in Mocktail"));
  std::string error;
  const std::vector<ChangelogEntry> entries = LoadChangelog(&error);
  if (entries.empty()) {
    GtkWidget* group = AddGroup(
        page, _("Changelog"),
        error.empty() ? _("Nothing here yet") : error.c_str());
    GtkWidget* row =
        NewActionRow("book-symbolic", [] {});
    RowSpec spec;
    spec.title = _("No release notes found");
    spec.keywords = {"changelog", "changes", "новости", "изменения",
                     "версии", "журнал"};
    spec.hint.subtitle =
        _("Ask the maintainer to ship a changelog.json with the build");
    AddRow(group, DecorateRow(context, row, std::move(spec)));
    return page;
  }
  for (const ChangelogEntry& entry : entries) {
    BuildEntryGroup(page, entry);
  }
  return page;
}

}  // namespace mocktail::launcher_ui
