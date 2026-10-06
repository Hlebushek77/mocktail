#include "mocktail/graphics/vulkan_etc2_caps.h"

#include <gtest/gtest.h>

namespace mocktail::graphics {
namespace {

TEST(VulkanEtc2CapsTest, FormatsAdvertisedTruthTable) {
  struct Row {
    bool emulated;
    bool advertised;
    bool formats;
  };
  constexpr Row kRows[] = {
      // A device with native ETC2 (or emulation off) is never rewritten.
      {false, false, false},
      {false, true, false},
      // The default: ETC2 advertised through the emulation.
      {true, true, true},
      // MOCKTAIL_ADVERTISE_ETC2=0: the formats follow the hidden feature.
      {true, false, false},
  };
  for (const Row& row : kRows) {
    EXPECT_EQ(Etc2FormatsAdvertised(row.emulated, row.advertised),
              row.formats)
        << "emulated=" << row.emulated << " advertised=" << row.advertised;
  }
}

TEST(VulkanEtc2CapsTest, FormatsAlwaysMatchTheFeatureBit) {
  for (const bool emulated : {false, true}) {
    for (const bool advertised : {false, true}) {
      EXPECT_EQ(Etc2FormatsAdvertised(emulated, advertised),
                Etc2FeatureAdvertised(emulated, advertised))
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

}  // namespace
}  // namespace mocktail::graphics
