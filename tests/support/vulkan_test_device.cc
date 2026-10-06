#include "support/vulkan_test_device.h"

#include <dirent.h>
#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

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

std::mutex& ValidationMutex() {
  static auto* mutex = new std::mutex;
  return *mutex;
}

std::vector<std::string>& ValidationMessages() {
  static auto* messages = new std::vector<std::string>;
  return *messages;
}

VKAPI_ATTR VkBool32 VKAPI_CALL
CollectValidation(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                  VkDebugUtilsMessageTypeFlagsEXT types,
                  const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
  if ((types & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT) == 0 ||
      severity < VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT ||
      data == nullptr) {
    return VK_FALSE;
  }
  std::string message =
      severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ? "ERROR "
                                                                : "WARNING ";
  message += data->pMessageIdName != nullptr ? data->pMessageIdName : "?";
  message += ": ";
  message += data->pMessage != nullptr ? data->pMessage : "";
  std::lock_guard<std::mutex> lock(ValidationMutex());
  ValidationMessages().push_back(std::move(message));
  return VK_FALSE;
}

bool HasLayer(PFN_vkGetInstanceProcAddr get_instance_proc_addr,
              const char* name) {
  auto enumerate = reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(
      get_instance_proc_addr(VK_NULL_HANDLE,
                             "vkEnumerateInstanceLayerProperties"));
  std::uint32_t count = 0;
  if (enumerate == nullptr || enumerate(&count, nullptr) != VK_SUCCESS) {
    return false;
  }
  std::vector<VkLayerProperties> layers(count);
  enumerate(&count, layers.data());
  layers.resize(count);
  return std::any_of(layers.begin(), layers.end(),
                     [name](const VkLayerProperties& layer) {
                       return std::strcmp(layer.layerName, name) == 0;
                     });
}

bool LayerHasExtension(PFN_vkGetInstanceProcAddr get_instance_proc_addr,
                       const char* layer, const char* extension) {
  auto enumerate =
      reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
          get_instance_proc_addr(VK_NULL_HANDLE,
                                 "vkEnumerateInstanceExtensionProperties"));
  std::uint32_t count = 0;
  if (enumerate == nullptr ||
      enumerate(layer, &count, nullptr) != VK_SUCCESS) {
    return false;
  }
  std::vector<VkExtensionProperties> extensions(count);
  enumerate(layer, &count, extensions.data());
  extensions.resize(count);
  return std::any_of(extensions.begin(), extensions.end(),
                     [extension](const VkExtensionProperties& candidate) {
                       return std::strcmp(candidate.extensionName,
                                          extension) == 0;
                     });
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
  return Create(skip_reason, Options());
}

std::unique_ptr<HeadlessVulkanDevice> HeadlessVulkanDevice::Create(
    std::string* skip_reason, const Options& options) {
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
  // MOCKTAIL_TEST_VALIDATION: the Khronos layer, enabled here rather than
  // through VK_INSTANCE_LAYERS, so the messenger below sees every message.
  constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";
  const char* validation = std::getenv("MOCKTAIL_TEST_VALIDATION");
  const bool validate = validation != nullptr && validation[0] != '\0' &&
                        std::strcmp(validation, "0") != 0;
  const bool validate_sync =
      validate && std::strcmp(validation, "core") != 0;
  std::vector<const char*> instance_extensions;
  VkValidationFeatureEnableEXT sync_feature =
      VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
  VkValidationFeaturesEXT features{};
  features.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
  if (validate) {
    if (!HasLayer(get_instance_proc_addr, kValidationLayer) ||
        !LayerHasExtension(get_instance_proc_addr, kValidationLayer,
                           VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
      *skip_reason =
          "MOCKTAIL_TEST_VALIDATION is set, but VK_LAYER_KHRONOS_validation "
          "(with VK_EXT_debug_utils) is not installed";
      return nullptr;
    }
    instance_info.enabledLayerCount = 1;
    instance_info.ppEnabledLayerNames = &kValidationLayer;
    instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (validate_sync) {
      if (LayerHasExtension(get_instance_proc_addr, kValidationLayer,
                            VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME)) {
        instance_extensions.push_back(
            VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
      }
      features.enabledValidationFeatureCount = 1;
      features.pEnabledValidationFeatures = &sync_feature;
      instance_info.pNext = &features;
    }
    instance_info.enabledExtensionCount =
        static_cast<std::uint32_t>(instance_extensions.size());
    instance_info.ppEnabledExtensionNames = instance_extensions.data();
  }
  const VkResult instance_result =
      create_instance(&instance_info, nullptr, &device->instance_);
  if (instance_result != VK_SUCCESS) {
    *skip_reason = "no Vulkan driver: vkCreateInstance returned " +
                   std::to_string(instance_result);
    return nullptr;
  }
  if (validate) {
    auto create_messenger =
        reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            get_instance_proc_addr(device->instance_,
                                   "vkCreateDebugUtilsMessengerEXT"));
    VkDebugUtilsMessengerCreateInfoEXT messenger_info{};
    messenger_info.sType =
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    messenger_info.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    messenger_info.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    messenger_info.pfnUserCallback = CollectValidation;
    if (create_messenger == nullptr ||
        create_messenger(device->instance_, &messenger_info, nullptr,
                         &device->messenger_) != VK_SUCCESS) {
      *skip_reason = "could not create a VK_EXT_debug_utils messenger";
      return nullptr;
    }
    device->sync_validation_ = validate_sync;
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
  for (std::uint32_t index = 0; index < count; ++index) {
    const VkQueueFlags flags = device->queue_families_[index].queueFlags;
    if (index == family || device->queue_families_[index].queueCount == 0) {
      continue;
    }
    if (options.compute_only_family &&
        device->compute_only_family_ == UINT32_MAX &&
        (flags & VK_QUEUE_COMPUTE_BIT) != 0 &&
        (flags & VK_QUEUE_GRAPHICS_BIT) == 0) {
      device->compute_only_family_ = index;
    }
    if (options.transfer_only_family &&
        device->transfer_only_family_ == UINT32_MAX &&
        (flags & VK_QUEUE_TRANSFER_BIT) != 0 &&
        (flags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == 0) {
      device->transfer_only_family_ = index;
    }
  }

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
  // Low priority: stay out of the desktop's way.
  const std::uint32_t main_queues = std::max<std::uint32_t>(
      1, std::min(options.queue_count,
                  device->queue_families_[family].queueCount));
  const std::vector<float> priorities(main_queues, 0.0f);
  std::vector<VkDeviceQueueCreateInfo> queue_infos;
  const auto add_family = [&](std::uint32_t queue_family,
                              std::uint32_t queues) {
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = queue_family;
    queue_info.queueCount = queues;
    queue_info.pQueuePriorities = priorities.data();
    queue_infos.push_back(queue_info);
  };
  add_family(family, main_queues);
  for (const std::uint32_t extra :
       {device->compute_only_family_, device->transfer_only_family_}) {
    if (extra != UINT32_MAX) {
      add_family(extra, 1);
    }
  }
  VkDeviceCreateInfo device_info{};
  device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_info.pNext = synchronization2 ? &vulkan13 : nullptr;
  device_info.queueCreateInfoCount =
      static_cast<std::uint32_t>(queue_infos.size());
  device_info.pQueueCreateInfos = queue_infos.data();
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
  for (const VkDeviceQueueCreateInfo& queue_info : queue_infos) {
    for (std::uint32_t index = 0; index < queue_info.queueCount; ++index) {
      VkQueue queue = VK_NULL_HANDLE;
      vk.vkGetDeviceQueue(device->device_, queue_info.queueFamilyIndex, index,
                          &queue);
      device->queues_.push_back(
          {{queue_info.queueFamilyIndex, index}, queue});
    }
  }
  device->queue_ = device->Queue(family, 0);
  return device;
}

VkQueue HeadlessVulkanDevice::Queue(std::uint32_t family,
                                    std::uint32_t index) const {
  for (const auto& [key, queue] : queues_) {
    if (key.first == family && key.second == index) {
      return queue;
    }
  }
  return VK_NULL_HANDLE;
}

std::vector<std::string> HeadlessVulkanDevice::TakeValidationMessages() {
  std::lock_guard<std::mutex> lock(ValidationMutex());
  std::vector<std::string> messages;
  messages.swap(ValidationMessages());
  return messages;
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
  if (messenger_ != VK_NULL_HANDLE) {
    auto destroy_messenger =
        reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            get_instance_proc_addr_(instance_,
                                    "vkDestroyDebugUtilsMessengerEXT"));
    if (destroy_messenger != nullptr) {
      destroy_messenger(instance_, messenger_, nullptr);
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
         std::to_string(queue_family_) +
         (validating() ? (sync_validation_ ? ", validation with sync"
                                           : ", validation")
                       : "");
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
