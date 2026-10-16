// See ui_prefs.h.
#include "launcher_ui/ui_prefs.h"

#include <adwaita.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

#include "launcher_ui/pages.h"

namespace mocktail::launcher_ui {
namespace {

std::filesystem::path ConfigDir() {
  const char* xdg = std::getenv("XDG_CONFIG_HOME");
  std::filesystem::path base =
      (xdg != nullptr && xdg[0] != '\0')
          ? xdg
          : std::filesystem::path(std::getenv("HOME") ? std::getenv("HOME")
                                                       : "/root") /
                ".config";
  return base / "mocktail";
}

std::string StrField(const nlohmann::json& object, const char* key,
                     const std::string& fallback = {}) {
  auto found = object.find(key);
  if (found == object.end() || !found->is_string()) return fallback;
  return found->get<std::string>();
}

}  // namespace

std::string UiPrefsPath() {
  return (ConfigDir() / "launcher.json").string();
}

UiPrefs LoadUiPrefs() {
  UiPrefs prefs;
  std::ifstream file(UiPrefsPath());
  if (!file) return prefs;
  nlohmann::json document =
      nlohmann::json::parse(file, nullptr, /*allow_exceptions=*/false);
  if (document.is_discarded() || !document.is_object()) return prefs;
  prefs.theme = StrField(document, "theme", "system");
  prefs.accent = StrField(document, "accent", "default");
  prefs.language = StrField(document, "language");
  auto strings = [&](const char* key) {
    std::vector<std::string> out;
    auto found = document.find(key);
    if (found == document.end() || !found->is_array()) return out;
    for (const auto& item : *found) {
      if (item.is_string() && out.size() < 64) {
        out.push_back(item.get<std::string>());
      }
    }
    return out;
  };
  prefs.tab_order = strings("tab_order");
  prefs.hidden_tabs = strings("hidden_tabs");
  if (prefs.theme != "dark" && prefs.theme != "light") prefs.theme = "system";
  return prefs;
}

bool SaveUiPrefs(const UiPrefs& prefs) {
  nlohmann::json document = nlohmann::json::object();
  document["theme"] = prefs.theme;
  document["accent"] = prefs.accent;
  document["language"] = prefs.language;
  document["tab_order"] = prefs.tab_order;
  document["hidden_tabs"] = prefs.hidden_tabs;
  std::error_code error;
  std::filesystem::create_directories(ConfigDir(), error);
  std::ofstream file(UiPrefsPath());
  if (!file) return false;
  file << document.dump(2);
  return static_cast<bool>(file);
}

std::vector<const char*> OrderedTabIds() {
  const UiPrefs prefs = LoadUiPrefs();
  std::vector<const char*> ordered;
  std::vector<bool> used(Sections().size(), false);
  auto find = [&](const std::string& id) -> const char* {
    for (std::size_t i = 0; i < Sections().size(); ++i) {
      if (!used[i] && Sections()[i].id == id) {
        used[i] = true;
        return Sections()[i].id;
      }
    }
    return nullptr;
  };
  for (const std::string& id : prefs.tab_order) {
    if (const char* hit = find(id)) ordered.push_back(hit);
  }
  for (std::size_t i = 0; i < Sections().size(); ++i) {
    if (!used[i]) {
      used[i] = true;
      ordered.push_back(Sections()[i].id);
    }
  }
  // Hidden tabs (never Extras itself) are dropped by the builders.
  std::vector<const char*> visible;
  for (const char* id : ordered) {
    if (std::string(id) == "extras") {
      visible.push_back(id);
      continue;
    }
    bool hidden = false;
    for (const std::string& h : prefs.hidden_tabs) {
      if (h == id) {
        hidden = true;
        break;
      }
    }
    if (!hidden) visible.push_back(id);
  }
  return visible;
}

void ApplyUiTheme() {
  const UiPrefs prefs = LoadUiPrefs();
  AdwStyleManager* manager = adw_style_manager_get_default();
  if (prefs.theme == "dark") {
    adw_style_manager_set_color_scheme(manager, ADW_COLOR_SCHEME_FORCE_DARK);
  } else if (prefs.theme == "light") {
    adw_style_manager_set_color_scheme(manager, ADW_COLOR_SCHEME_FORCE_LIGHT);
  } else {
    adw_style_manager_set_color_scheme(manager, ADW_COLOR_SCHEME_DEFAULT);
  }
}

void ApplyUiAccent() {
  const UiPrefs prefs = LoadUiPrefs();
  static GtkCssProvider* provider = nullptr;
  if (provider != nullptr) {
    gtk_style_context_remove_provider_for_display(
        gdk_display_get_default(), GTK_STYLE_PROVIDER(provider));
    g_object_unref(provider);
    provider = nullptr;
  }
  const char* background = nullptr;
  const char* foreground = "white";
  if (prefs.accent == "blue") {
    background = "#3584e4";
  } else if (prefs.accent == "green") {
    background = "#2ec27e";
    foreground = "black";
  } else if (prefs.accent == "orange") {
    background = "#ff7800";
  } else if (prefs.accent == "purple") {
    background = "#9141ac";
  } else if (prefs.accent == "red") {
    background = "#e01b24";
  }
  if (background == nullptr) return;
  char css[256];
  snprintf(css, sizeof(css),
           "@define-color accent_bg_color %s;\n"
           "@define-color accent_fg_color %s;\n",
           background, foreground);
  provider = gtk_css_provider_new();
  gtk_css_provider_load_from_string(provider, css);
  gtk_style_context_add_provider_for_display(
      gdk_display_get_default(), GTK_STYLE_PROVIDER(provider),
      GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
}

}  // namespace mocktail::launcher_ui
