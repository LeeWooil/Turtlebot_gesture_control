// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from custom_interfaces:srv/TurningControl.idl
// generated code does not contain a copyright notice
#include "custom_interfaces/srv/detail/turning_control__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "custom_interfaces/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "custom_interfaces/srv/detail/turning_control__struct.h"
#include "custom_interfaces/srv/detail/turning_control__functions.h"
#include "fastcdr/Cdr.h"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif


// forward declare type support functions


using _TurningControl_Request__ros_msg_type = custom_interfaces__srv__TurningControl_Request;

static bool _TurningControl_Request__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _TurningControl_Request__ros_msg_type * ros_message = static_cast<const _TurningControl_Request__ros_msg_type *>(untyped_ros_message);
  // Field name: time_duration
  {
    cdr << ros_message->time_duration;
  }

  // Field name: angular_vel_z
  {
    cdr << ros_message->angular_vel_z;
  }

  // Field name: linear_vel_x
  {
    cdr << ros_message->linear_vel_x;
  }

  return true;
}

static bool _TurningControl_Request__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _TurningControl_Request__ros_msg_type * ros_message = static_cast<_TurningControl_Request__ros_msg_type *>(untyped_ros_message);
  // Field name: time_duration
  {
    cdr >> ros_message->time_duration;
  }

  // Field name: angular_vel_z
  {
    cdr >> ros_message->angular_vel_z;
  }

  // Field name: linear_vel_x
  {
    cdr >> ros_message->linear_vel_x;
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_custom_interfaces
size_t get_serialized_size_custom_interfaces__srv__TurningControl_Request(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _TurningControl_Request__ros_msg_type * ros_message = static_cast<const _TurningControl_Request__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name time_duration
  {
    size_t item_size = sizeof(ros_message->time_duration);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name angular_vel_z
  {
    size_t item_size = sizeof(ros_message->angular_vel_z);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name linear_vel_x
  {
    size_t item_size = sizeof(ros_message->linear_vel_x);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _TurningControl_Request__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_custom_interfaces__srv__TurningControl_Request(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_custom_interfaces
size_t max_serialized_size_custom_interfaces__srv__TurningControl_Request(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;

  // member: time_duration
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: angular_vel_z
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: linear_vel_x
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  return current_alignment - initial_alignment;
}

static size_t _TurningControl_Request__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_custom_interfaces__srv__TurningControl_Request(
    full_bounded, 0);
}


static message_type_support_callbacks_t __callbacks_TurningControl_Request = {
  "custom_interfaces::srv",
  "TurningControl_Request",
  _TurningControl_Request__cdr_serialize,
  _TurningControl_Request__cdr_deserialize,
  _TurningControl_Request__get_serialized_size,
  _TurningControl_Request__max_serialized_size
};

static rosidl_message_type_support_t _TurningControl_Request__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_TurningControl_Request,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, custom_interfaces, srv, TurningControl_Request)() {
  return &_TurningControl_Request__type_support;
}

#if defined(__cplusplus)
}
#endif

// already included above
// #include <cassert>
// already included above
// #include <limits>
// already included above
// #include <string>
// already included above
// #include "rosidl_typesupport_fastrtps_c/identifier.h"
// already included above
// #include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
// already included above
// #include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
// already included above
// #include "custom_interfaces/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
// already included above
// #include "custom_interfaces/srv/detail/turning_control__struct.h"
// already included above
// #include "custom_interfaces/srv/detail/turning_control__functions.h"
// already included above
// #include "fastcdr/Cdr.h"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif


// forward declare type support functions


using _TurningControl_Response__ros_msg_type = custom_interfaces__srv__TurningControl_Response;

static bool _TurningControl_Response__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _TurningControl_Response__ros_msg_type * ros_message = static_cast<const _TurningControl_Response__ros_msg_type *>(untyped_ros_message);
  // Field name: success
  {
    cdr << (ros_message->success ? true : false);
  }

  return true;
}

static bool _TurningControl_Response__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _TurningControl_Response__ros_msg_type * ros_message = static_cast<_TurningControl_Response__ros_msg_type *>(untyped_ros_message);
  // Field name: success
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->success = tmp ? true : false;
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_custom_interfaces
size_t get_serialized_size_custom_interfaces__srv__TurningControl_Response(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _TurningControl_Response__ros_msg_type * ros_message = static_cast<const _TurningControl_Response__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name success
  {
    size_t item_size = sizeof(ros_message->success);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _TurningControl_Response__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_custom_interfaces__srv__TurningControl_Response(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_custom_interfaces
size_t max_serialized_size_custom_interfaces__srv__TurningControl_Response(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;

  // member: success
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }

  return current_alignment - initial_alignment;
}

static size_t _TurningControl_Response__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_custom_interfaces__srv__TurningControl_Response(
    full_bounded, 0);
}


static message_type_support_callbacks_t __callbacks_TurningControl_Response = {
  "custom_interfaces::srv",
  "TurningControl_Response",
  _TurningControl_Response__cdr_serialize,
  _TurningControl_Response__cdr_deserialize,
  _TurningControl_Response__get_serialized_size,
  _TurningControl_Response__max_serialized_size
};

static rosidl_message_type_support_t _TurningControl_Response__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_TurningControl_Response,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, custom_interfaces, srv, TurningControl_Response)() {
  return &_TurningControl_Response__type_support;
}

#if defined(__cplusplus)
}
#endif

#include "rosidl_typesupport_fastrtps_cpp/service_type_support.h"
#include "rosidl_typesupport_cpp/service_type_support.hpp"
// already included above
// #include "rosidl_typesupport_fastrtps_c/identifier.h"
// already included above
// #include "custom_interfaces/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "custom_interfaces/srv/turning_control.h"

#if defined(__cplusplus)
extern "C"
{
#endif

static service_type_support_callbacks_t TurningControl__callbacks = {
  "custom_interfaces::srv",
  "TurningControl",
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, custom_interfaces, srv, TurningControl_Request)(),
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, custom_interfaces, srv, TurningControl_Response)(),
};

static rosidl_service_type_support_t TurningControl__handle = {
  rosidl_typesupport_fastrtps_c__identifier,
  &TurningControl__callbacks,
  get_service_typesupport_handle_function,
};

const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, custom_interfaces, srv, TurningControl)() {
  return &TurningControl__handle;
}

#if defined(__cplusplus)
}
#endif
