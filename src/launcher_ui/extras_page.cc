// Extras page ("Дополнительно"): launcher-local customization that never
// touches config.yaml (the game rejects unknown keys). Backed by
// launcher.json via ui_prefs.h. Theme and accent apply live; language and
// tab layout apply on restart.
#include <adwaita.h>

#include <algorithm>
#include <string>
#include <vector>

#include "launcher_ui/bindings.h"
#include "launcher_ui/i18n.h"
#include "launcher_ui/launcher_context.h"
#include "launcher_ui/pages.h"
#include "launcher_ui/ui_prefs.h"

namespace mocktail::launcher_ui {
namespace {

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

int IndexOf(const std::string& value,
            const std::vector<std::string>& values) {
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (values[i] == value) return static_cast<int>(i);
  }
  return 0;
}

GtkWidget* BuildThemeRow(LauncherContext* context) {
  UiPrefs prefs = LoadUiPrefs();
  const std::vector<std::string> values = {"system", "dark", "light"};
  const std::vector<std::string> labels = {
      _("System"), _("Dark"), _("Light")};
  GtkWidget* row = MakeCombo(labels, IndexOf(prefs.theme, values));
  RowSpec spec;
  spec.title = _("Theme");
  spec.keywords = {"theme", "dark", "light", "тема", "тёмная", "светлая",
                   "оформление"};
  spec.hint.subtitle = _("Applies immediately");
  g_signal_connect(
      row, "notify::selected",
      G_CALLBACK(+[](GObject* object, GParamSpec*, gpointer data) {
        auto* self = static_cast<LauncherContext*>(data);
        (void)self;
        guint selected =
            adw_combo_row_get_selected(ADW_COMBO_ROW(object));
        const char* values[] = {"system", "dark", "light"};
        if (selected > 2) return;
        UiPrefs prefs = LoadUiPrefs();
        prefs.theme = values[selected];
        if (SaveUiPrefs(prefs)) ApplyUiTheme();
      }),
      context);
  return DecorateRow(context, row, std::move(spec));
}

GtkWidget* BuildAccentRow(LauncherContext* context) {
  UiPrefs prefs = LoadUiPrefs();
  const std::vector<std::string> values = {"default", "blue",  "green",
                                           "orange",  "purple", "red"};
  const std::vector<std::string> labels = {
      _("Default"), _("Blue"),  _("Green"),
      _("Orange"),  _("Purple"), _("Red")};
  GtkWidget* row = MakeCombo(labels, IndexOf(prefs.accent, values));
  RowSpec spec;
  spec.title = _("Accent color");
  spec.keywords = {"accent", "color", "акцент", "цвет"};
  spec.hint.subtitle = _("Applies immediately");
  g_signal_connect(
      row, "notify::selected",
      G_CALLBACK(+[](GObject* object, GParamSpec*, gpointer data) {
        auto* self = static_cast<LauncherContext*>(data);
        (void)self;
        guint selected =
            adw_combo_row_get_selected(ADW_COMBO_ROW(object));
        const char* values[] = {"default", "blue",  "green",
                                "orange",  "purple", "red"};
        if (selected > 5) return;
        UiPrefs prefs = LoadUiPrefs();
        prefs.accent = values[selected];
        if (SaveUiPrefs(prefs)) ApplyUiAccent();
      }),
      context);
  return DecorateRow(context, row, std::move(spec));
}

GtkWidget* BuildLanguageRow(LauncherContext* context) {
  UiPrefs prefs = LoadUiPrefs();
  const std::vector<std::string> values = {"", "en", "ru"};
  const std::vector<std::string> labels = {
      _("System language"), _("English"), "Русский"};
  GtkWidget* row = MakeCombo(labels, IndexOf(prefs.language, values));
  RowSpec spec;
  spec.title = _("Interface language");
  spec.keywords = {"language", "locale", "язык", "интерфейс", "перевод"};
  spec.hint.subtitle = _("Applies on restart");
  g_signal_connect(
      row, "notify::selected",
      G_CALLBACK(+[](GObject* object, GParamSpec*, gpointer data) {
        auto* self = static_cast<LauncherContext*>(data);
        guint selected =
            adw_combo_row_get_selected(ADW_COMBO_ROW(object));
        const char* values[] = {"", "en", "ru"};
        if (selected > 2) return;
        UiPrefs prefs = LoadUiPrefs();
        prefs.language = values[selected];
        if (SaveUiPrefs(prefs) && self != nullptr) {
          self->Toast(_("Restart the launcher to apply the language"));
        }
      }),
      context);
  return DecorateRow(context, row, std::move(spec));
}

// One row per sidebar section: visibility switch + up/down order buttons.
// Applies on restart (_sidebar and pages are built once at startup).
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
  // Current order first, then anything new.
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
    // Capture row index by value; order/visibility persist on restart.
    struct TabCtl {
      LauncherContext* context;
      std::size_t row;
    };
    auto* up_ctl = new TabCtl{context, row};
    g_signal_connect(
        up, "clicked",
        G_CALLBACK(+[](GtkButton*, gpointer data) {
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
        }),
        up_ctl);
    auto* down_ctl = new TabCtl{context, row};
    g_signal_connect(
        down, "clicked",
        G_CALLBACK(+[](GtkButton*, gpointer data) {
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
        }),
        down_ctl);
    struct HideCtl {
      LauncherContext* context;
      std::string id;
    };
    auto* hide_ctl = new HideCtl{context, info.id};
    g_signal_connect(
        toggle, "notify::active",
        G_CALLBACK(+[](GObject* object, GParamSpec*, gpointer data) {
          std::unique_ptr<HideCtl> ctl(static_cast<HideCtl*>(data));
          UiPrefs prefs = LoadUiPrefs();
          const bool active =
              gtk_switch_get_active(GTK_SWITCH(object)) != FALSE;
          auto& hidden = prefs.hidden_tabs;
          hidden.erase(
              std::remove(hidden.begin(), hidden.end(), ctl->id),
              hidden.end());
          if (!active) hidden.push_back(ctl->id);
          SaveUiPrefs(prefs);
        }),
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
  GtkWidget* lang =
      AddGroup(page, _("Language"), _("Applies on restart"));
  AddRow(lang, BuildLanguageRow(context));
  BuildTabsGroup(page, context);
  return page;
}

}  // namespace mocktail::launcher_ui
