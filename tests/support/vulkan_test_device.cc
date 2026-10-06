#include "support/vulkan_test_device.h"

#include <dirent.h>
#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace mocktail::graphics::testing {
namespace {

// The process name in /proc/<pid>/comm, or "" when it cannot be read.
std::string ProcessName(long pid) {
  const std::string path = "/proc/" + std::to_string(pid) + "/comm";
  std::FILE* file = std::fopen(path.c_str(), "r");
  if (file == nullptr) {
    return {};
  }
  char name[64] = {};
  const bool read = std::fgets(name, sizeof(name), file) != nullptr;
  std::fclose(file);
  if (!read) {
    return {};
  }
  name[std::strcspn(name, "\n")] = '\0';
  return name;
}

// The parent from /proc/<pid>/stat, or 0. The name in the second field may
// hold spaces and parentheses, so the fields are read after the last ')'.
long ParentProcess(long pid) {
  const std::string path = "/proc/" + std::to_string(pid) + "/stat";
  std::FILE* file = std::fopen(path.c_str(), "r");
  if (file == nullptr) {
    return 0;
  }
  char line[512] = {};
  const bool read = std::fgets(line, sizeof(line), file) != nullptr;
  std::fclose(file);
  const char* name_end = read ? std::strrchr(line, ')') : nullptr;
  char state = 0;
  long parent = 0;
  if (name_end == nullptr ||
      std::sscanf(name_end + 1, " %c %ld", &state, &parent) != 2) {
    return 0;
  }
  return parent;
}

// ctest runs the whole suite in parallel, and
// AnyLinuxPackagingTest.RelocatedRuntimePaths runs a fixture binary named
// mocktail: without this, every GPU test that starts during those two
// seconds skips as if a game were running. A process that descends from
// the ctest which started this test is another test, never a live game.
bool StartedByThisCtestRun(long pid) {
  const long ctest = static_cast<long>(getppid());
  if (ctest <= 1 || ProcessName(ctest) != "ctest") {
    return false;
  }
  for (int depth = 0; depth < 64 && pid > 1; ++depth) {
    pid = ParentProcess(pid);
    if (pid == ctest) {
      return true;
    }
  }
  return false;
}

// The loader stays loaded for the life of the process: unloading drivers at
// exit is a known source of crashes, and nothing is gained by it.
PFN_vkGetInstanceProcAddr LoadVulkanLoader(std::string* why) {
  static void* library = nullptr;
  static std::string error;
  static bool tried = false;
  if (!tried) {
    tried = true;
    library = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
      const char* message = dlerror();
      error = message != nullptr ? message : "unknown error";
    }
  }
  if (library == nullptr) {
    *why = "no Vulkan loader: " + error;
    return nullptr;
  }
  auto proc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
      dlsym(library, "vkGetInstanceProcAddr"));
  if (proc == nullptr) {
    *why = "the Vulkan loader exports no vkGetInstanceProcAddr";
  }
  return proc;
}

int TypeRank(VkPhysicalDeviceType type) {
  switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
      return 0;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
      return 1;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
      return 2;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
      return 3;
    default:
      return 4;
  }
}

std::string VersionText(std::uint32_t vendor, std::uint32_t version) {
  char text[64];
  if (vendor == 0x10de) {  // NVIDIA packs 10.8.8.6 bits
    std::snprintf(text, sizeof(text), "%u.%u.%u", version >> 22,
                  (version >> 14) & 0xff, (version >> 6) & 0xff);
  } else {
    std::snprintf(text, sizeof(text), "%u.%u.%u", VK_API_VERSION_MAJOR(version),
                  VK_API_VERSION_MINOR(version), VK_API_VERSION_PATCH(version));
  }
  return text;
}

}  // namespace

bool LiveGameRunning(std::string* description) {
  DIR* proc = opendir("/proc");
  if (proc == nullptr) {
    return false;
  }
  bool found = false;
  while (const dirent* entry = readdir(proc)) {
    const char* pid = entry->d_name;
    if (pid[0] == '\0' || !std::all_of(pid, pid + std::strlen(pid), [](char c) {
          return std::isdigit(static_cast<unsigned char>(c)) != 0;
        })) {
      continue;
    }
    const long number = std::strtol(pid, nullptr, 10);
    const std::string name = ProcessName(number);
    if ((name == "mocktail" || name == "sober") &&
        !StartedByThisCtestRun(number)) {
      if (description != nullptr) {
        *description = name + " (pid " + pid + ")";
      }
      found = true;
      break;
    }
  }
  closedir(proc);
  return found;
}

std::unique_ptr<HeadlessVulkanDevice> HeadlessVulkanDevice::Create(
    std::string* skip_reason) {
  std::string why;
  std::string live;
  if (LiveGameRunning(&live)) {
    *skip_reason = "a live game is running (" + live +
                   "); GPU tests stay off the GPU while it is measured "
                   "(CR-25)";
    return nullptr;
  }
  const PFN_vkGetInstanceProcAddr get_instance_proc_addr =
      LoadVulkanLoader(&why);
  if (get_instance_proc_addr == nullptr) {
    *skip_reason = why;
    return nullptr;
  }
  auto create_instance = reinterpret_cast<PFN_vkCreateInstance>(
      get_instance_proc_addr(VK_NULL_HANDLE, "vkCreateInstance"));
  auto enumerate_version = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
      get_instance_proc_addr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
  if (create_instance == nullptr) {
    *skip_reason = "the Vulkan loader has no vkCreateInstance";
    return nullptr;
  }
  std::uint32_t api_version = VK_API_VERSION_1_0;
  if (enumerate_version != nullptr) {
    enumerate_version(&api_version);
  }
  api_version = std::min<std::uint32_t>(api_version, VK_API_VERSION_1_3);

  std::unique_ptr<HeadlessVulkanDevice> device(new HeadlessVulkanDevice);
  device->get_instance_proc_addr_ = get_instance_proc_addr;
  VkApplicationInfo application{};
  application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  application.pApplicationName = "mocktail-gpu-test";
  application.apiVersion = api_version;
  VkInstanceCreateInfo instance_info{};
  instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instance_info.pApplicationInfo = &application;
  const VkResult instance_result =
      create_instance(&instance_info, nullptr, &device->instance_);
  if (instance_result != VK_SUCCESS) {
    *skip_reason = "no Vulkan driver: vkCreateInstance returned " +
                   std::to_string(instance_result);
    return nullptr;
  }
  Dispatch& vk = device->vk_;
#define MOCKTAIL_VULKAN_TEST_LOAD(name)                  \
  vk.name = reinterpret_cast<PFN_##name>(                \
      get_instance_proc_addr(device->instance_, #name)); \
  if (vk.name == nullptr) {                              \
    *skip_reason = "the Vulkan loader has no " #name;    \
    return nullptr;                                      \
  }
  MOCKTAIL_VULKAN_TEST_INSTANCE_FUNCTIONS(MOCKTAIL_VULKAN_TEST_LOAD)
#undef MOCKTAIL_VULKAN_TEST_LOAD
#define MOCKTAIL_VULKAN_TEST_LOAD(name)   \
  vk.name = reinterpret_cast<PFN_##name>( \
      get_instance_proc_addr(device->instance_, #name));
  MOCKTAIL_VULKAN_TEST_OPTIONAL_INSTANCE_FUNCTIONS(MOCKTAIL_VULKAN_TEST_LOAD)
#undef MOCKTAIL_VULKAN_TEST_LOAD

  std::uint32_t count = 0;
  vk.vkEnumeratePhysicalDevices(device->instance_, &count, nullptr);
  std::vector<VkPhysicalDevice> physical_devices(count);
  if (count != 0) {
    vk.vkEnumeratePhysicalDevices(device->instance_, &count,
                                  physical_devices.data());
  }
  physical_devices.resize(count);
  if (physical_devices.empty()) {
    *skip_reason = "the Vulkan loader found no physical device";
    return nullptr;
  }
  const char* wanted = std::getenv("MOCKTAIL_TEST_VULKAN_DEVICE");
  int best_rank = 1 << 20;
  for (std::size_t index = 0; index < physical_devices.size(); ++index) {
    VkPhysicalDeviceProperties properties{};
    vk.vkGetPhysicalDeviceProperties(physical_devices[index], &properties);
    int rank = TypeRank(properties.deviceType);
    if (wanted != nullptr && wanted[0] != '\0') {
      const bool by_index =
          std::isdigit(static_cast<unsigned char>(wanted[0])) &&
          std::strtoul(wanted, nullptr, 10) == index;
      const bool by_name =
          std::strstr(properties.deviceName, wanted) != nullptr;
      rank = by_index || by_name ? -1 : rank + 100;
    }
    if (rank < best_rank) {
      best_rank = rank;
      device->physical_device_ = physical_devices[index];
      device->properties_ = properties;
    }
  }
  vk.vkGetPhysicalDeviceMemoryProperties(device->physical_device_,
                                         &device->memory_);
  vk.vkGetPhysicalDeviceQueueFamilyProperties(device->physical_device_, &count,
                                              nullptr);
  device->queue_families_.resize(count);
  vk.vkGetPhysicalDeviceQueueFamilyProperties(device->physical_device_, &count,
                                              device->queue_families_.data());
  device->queue_families_.resize(count);
  std::uint32_t family = UINT32_MAX;
  for (const VkQueueFlags wanted_flags :
       {VK_QUEUE_GRAPHICS_BIT, VK_QUEUE_COMPUTE_BIT, VK_QUEUE_TRANSFER_BIT}) {
    for (std::uint32_t index = 0; index < count && family == UINT32_MAX;
         ++index) {
      if ((device->queue_families_[index].queueFlags & wanted_flags) != 0 &&
          device->queue_families_[index].queueCount != 0) {
        family = index;
      }
    }
  }
  if (family == UINT32_MAX) {
    *skip_reason = "the device has no queue family that can copy";
    return nullptr;
  }
  device->queue_family_ = family;

  std::vector<const char*> extensions;
  const std::uint32_t device_api =
      std::min(api_version, device->properties_.apiVersion);
  if (device_api >= VK_API_VERSION_1_3) {
    device->copy_commands2_ = true;
  } else {
    std::uint32_t extension_count = 0;
    vk.vkEnumerateDeviceExtensionProperties(device->physical_device_, nullptr,
                                            &extension_count, nullptr);
    std::vector<VkExtensionProperties> available(extension_count);
    vk.vkEnumerateDeviceExtensionProperties(device->physical_device_, nullptr,
                                            &extension_count, available.data());
    for (const VkExtensionProperties& extension : available) {
      if (std::strcmp(extension.extensionName,
                      VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME) == 0) {
        extensions.push_back(VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME);
        device->copy_commands2_ = true;
      }
    }
  }
  // vkQueueSubmit2 needs the synchronization2 feature (Vulkan 1.3).
  VkPhysicalDeviceVulkan13Features vulkan13{};
  vulkan13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
  bool synchronization2 = false;
  if (device_api >= VK_API_VERSION_1_3 &&
      vk.vkGetPhysicalDeviceFeatures2 != nullptr) {
    VkPhysicalDeviceFeatures2 features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &vulkan13;
    vk.vkGetPhysicalDeviceFeatures2(device->physical_device_, &features);
    synchronization2 = vulkan13.synchronization2 == VK_TRUE;
    vulkan13 = {};
    vulkan13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vulkan13.synchronization2 = synchronization2 ? VK_TRUE : VK_FALSE;
  }
  const float priority = 0.0f;  // stay out of the desktop's way
  VkDeviceQueueCreateInfo queue_info{};
  queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_info.queueFamilyIndex = family;
  queue_info.queueCount = 1;
  queue_info.pQueuePriorities = &priority;
  VkDeviceCreateInfo device_info{};
  device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_info.pNext = synchronization2 ? &vulkan13 : nullptr;
  device_info.queueCreateInfoCount = 1;
  device_info.pQueueCreateInfos = &queue_info;
  device_info.enabledExtensionCount =
      static_cast<std::uint32_t>(extensions.size());
  device_info.ppEnabledExtensionNames = extensions.data();
  const VkResult device_result = vk.vkCreateDevice(
      device->physical_device_, &device_info, nullptr, &device->device_);
  if (device_result != VK_SUCCESS) {
    *skip_reason = std::string("vkCreateDevice failed on ") +
                   device->properties_.deviceName + ": " +
                   std::to_string(device_result);
    return nullptr;
  }
#define MOCKTAIL_VULKAN_TEST_LOAD(name)                \
  vk.name = reinterpret_cast<PFN_##name>(              \
      vk.vkGetDeviceProcAddr(device->device_, #name)); \
  if (vk.name == nullptr) {                            \
    *skip_reason = "the device has no " #name;         \
    return nullptr;                                    \
  }
  MOCKTAIL_VULKAN_TEST_DEVICE_FUNCTIONS(MOCKTAIL_VULKAN_TEST_LOAD)
#undef MOCKTAIL_VULKAN_TEST_LOAD
  if (synchronization2) {
    vk.vkQueueSubmit2 = reinterpret_cast<PFN_vkQueueSubmit2>(
        vk.vkGetDeviceProcAddr(device->device_, "vkQueueSubmit2"));
  }
  vk.vkGetDeviceQueue(device->device_, family, 0, &device->queue_);
  return device;
}

HeadlessVulkanDevice::~HeadlessVulkanDevice() {
  if (device_ != VK_NULL_HANDLE) {
    if (vk_.vkDeviceWaitIdle != nullptr) {
      vk_.vkDeviceWaitIdle(device_);
    }
    if (vk_.vkDestroyDevice != nullptr) {
      vk_.vkDestroyDevice(device_, nullptr);
    }
  }
  if (instance_ != VK_NULL_HANDLE && vk_.vkDestroyInstance != nullptr) {
    vk_.vkDestroyInstance(instance_, nullptr);
  }
}

std::string HeadlessVulkanDevice::Describe() const {
  return std::string(properties_.deviceName) + " (driver " +
         VersionText(properties_.vendorID, properties_.driverVersion) +
         ", Vulkan " + VersionText(0, properties_.apiVersion) + "), family " +
         std::to_string(queue_family_);
}

std::uint32_t HeadlessVulkanDevice::FindMemoryType(
    std::uint32_t type_bits, VkMemoryPropertyFlags required,
    VkMemoryPropertyFlags preferred, VkMemoryPropertyFlags avoided) const {
  std::uint32_t fallback = UINT32_MAX;
  for (std::uint32_t index = 0; index < memory_.memoryTypeCount; ++index) {
    const VkMemoryPropertyFlags flags =
        memory_.memoryTypes[index].propertyFlags;
    if ((type_bits & (1U << index)) == 0 || (flags & required) != required) {
      continue;
    }
    if ((flags & preferred) == preferred && (flags & avoided) == 0) {
      return index;
    }
    if (fallback == UINT32_MAX) {
      fallback = index;
    }
  }
  return fallback;
}

bool HeadlessVulkanDevice::OptimalTilingSupports(
    VkFormat format, VkFormatFeatureFlags features) const {
  VkFormatProperties properties{};
  vk_.vkGetPhysicalDeviceFormatProperties(physical_device_, format,
                                          &properties);
  return (properties.optimalTilingFeatures & features) == features;
}

}  // namespace mocktail::graphics::testing
