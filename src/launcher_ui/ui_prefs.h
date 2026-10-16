// Launcher-local UI prefs (Extras tab): theme, accent, language,
// sidebar order/visibility. Lives in launcher.json next to config.yaml
// and never touches the runtime config schema (the game would reject
// unknown keys).
#ifndef MOCKTAIL_LAUNCHER_UI_UI_PREFS_H_
#define MOCKTAIL_LAUNCHER_UI_UI_PREFS_H_

#include <string>
#include <vector>

namespace mocktail::launcher_ui {

// Applies on restart except where noted live.
struct UiPrefs {
  // "system" | "dark" | "light" (theme applies live).
  std::string theme = "system";
  // accent name, "default" for none (applies live).
  std::string accent = "default";
  // "" = system locale, else "en", "ru", ... (restart).
  std::string language;
  // Section ids in sidebar order; unknown ids ignored, missing sections
  // appended (restart).
  std::vector<std::string> tab_order;
  // Hidden section ids (restart; Extras itself stays visible).
  std::vector<std::string> hidden_tabs;
};

std::string UiPrefsPath();
UiPrefs LoadUiPrefs();
bool SaveUiPrefs(const UiPrefs& prefs);

// Applies theme + accent right now (used at startup and on change).
void ApplyUiTheme();
void ApplyUiAccent();

// Ordered, visible sections for sidebar/pages loops.
std::vector<const char*> OrderedTabIds();

}  // namespace mocktail::launcher_ui

#endif  // MOCKTAIL_LAUNCHER_UI_UI_PREFS_H_
