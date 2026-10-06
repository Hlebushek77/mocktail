#include "mocktail/graphics/vulkan_etc2_caps.h"

#include <gtest/gtest.h>

#include <cstdlib>

namespace mocktail::graphics {
namespace {

constexpr Etc2FormatPropsMode kConsistent = Etc2FormatPropsMode::kConsistent;
constexpr Etc2FormatPropsMode kLegacy = Etc2FormatPropsMode::kLegacy;

TEST(VulkanEtc2CapsTest, FormatsAdvertisedTruthTable) {
  struct Row {
    bool emulated;
    bool advertised;
    Etc2FormatPropsMode mode;
    bool formats;
  };
  constexpr Row kRows[] = {
      // A device with native ETC2 (or emulation off) is never rewritten.
      {false, false, kConsistent, false},
      {false, true, kConsistent, false},
      {false, false, kLegacy, false},
      {false, true, kLegacy, false},
      // The default: ETC2 advertised through the emulation, in both modes.
      {true, true, kConsistent, true},
      {true, true, kLegacy, true},
      // MOCKTAIL_ADVERTISE_ETC2=0: the formats follow the hidden feature
      // unless the plus.1 behaviour is asked for.
      {true, false, kConsistent, false},
      {true, false, kLegacy, true},
  };
  for (const Row& row : kRows) {
    EXPECT_EQ(Etc2FormatsAdvertised(row.emulated, row.advertised, row.mode),
              row.formats)
        << "emulated=" << row.emulated << " advertised=" << row.advertised
        << " legacy=" << (row.mode == kLegacy);
  }
}

TEST(VulkanEtc2CapsTest, ConsistentFormatsAlwaysMatchTheFeatureBit) {
  for (const bool emulated : {false, true}) {
    for (const bool advertised : {false, true}) {
      EXPECT_EQ(Etc2FormatsAdvertised(emulated, advertised, kConsistent),
                Etc2FeatureAdvertised(emulated, advertised))
          << "emulated=" << emulated << " advertised=" << advertised;
    }
  }
}

TEST(VulkanEtc2CapsTest, LegacyModeKeepsThePlus1Answer) {
  // plus.1 keyed the format hooks on the emulated device alone.
  for (const bool emulated : {false, true}) {
    for (const bool advertised : {false, true}) {
      EXPECT_EQ(Etc2FormatsAdvertised(emulated, advertised, kLegacy), emulated)
          << "emulated=" << emulated << " advertised=" << advertised;
    }
  }
}

TEST(VulkanEtc2CapsTest, FeatureFollowsTheEmulationAndTheAdvertiseSwitch) {
  EXPECT_FALSE(Etc2FeatureAdvertised(false, false));
  EXPECT_FALSE(Etc2FeatureAdvertised(false, true));
  EXPECT_FALSE(Etc2FeatureAdvertised(true, false));
  EXPECT_TRUE(Etc2FeatureAdvertised(true, true));
}

TEST(VulkanEtc2CapsTest, ParsesTheLegacySwitch) {
  EXPECT_EQ(ParseEtc2FormatPropsMode(nullptr), kConsistent);
  EXPECT_EQ(ParseEtc2FormatPropsMode(""), kConsistent);
  EXPECT_EQ(ParseEtc2FormatPropsMode("consistent"), kConsistent);
  EXPECT_EQ(ParseEtc2FormatPropsMode("legacy"), kLegacy);
  EXPECT_EQ(ParseEtc2FormatPropsMode("LEGACY"), kConsistent);
  EXPECT_EQ(ParseEtc2FormatPropsMode("1"), kConsistent);
  EXPECT_EQ(ParseEtc2FormatPropsMode("legacy "), kConsistent);
}

TEST(VulkanEtc2CapsTest, ReadsTheLegacySwitchFromTheEnvironment) {
  // The mode is read once per process; ctest runs each test on its own.
  ASSERT_EQ(setenv("MOCKTAIL_ETC2_FORMAT_PROPS", "legacy", 1), 0);
  EXPECT_EQ(Etc2FormatPropsModeFromEnvironment(), kLegacy);
  ASSERT_EQ(unsetenv("MOCKTAIL_ETC2_FORMAT_PROPS"), 0);
  EXPECT_EQ(Etc2FormatPropsModeFromEnvironment(), kLegacy);
}

}  // namespace
}  // namespace mocktail::graphics
