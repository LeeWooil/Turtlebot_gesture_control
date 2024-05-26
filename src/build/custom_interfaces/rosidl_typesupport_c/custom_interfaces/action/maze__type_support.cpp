// generated from rosidl_typesupport_c/resource/idl__type_support.cpp.em
// with input from custom_interfaces:action/Maze.idl
// generated code does not contain a copyright notice

#include "cstddef"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "custom_interfaces/msg/rosidl_typesupport_c__visibility_control.h"
#include "custom_interfaces/action/detail/maze__struct.h"
#include "rosidl_typesupport_c/identifier.h"
#include "rosidl_typesupport_c/message_type_support_dispatch.h"
#include "rosidl_typesupport_c/type_support_map.h"
#include "rosidl_typesupport_c/visibility_control.h"
#include "rosidl_typesupport_interface/macros.h"

namespace custom_interfaces
{

namespace action
{

namespace rosidl_typesupport_c
{

typedef struct _Maze_Goal_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _Maze_Goal_type_support_ids_t;

static const _Maze_Goal_type_support_ids_t _Maze_Goal_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _Maze_Goal_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _Maze_Goal_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _Maze_Goal_type_support_symbol_names_t _Maze_Goal_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, custom_interfaces, action, Maze_Goal)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, custom_interfaces, action, Maze_Goal)),
  }
};

typedef struct _Maze_Goal_type_support_data_t
{
  void * data[2];
} _Maze_Goal_type_support_data_t;

static _Maze_Goal_type_support_data_t _Maze_Goal_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _Maze_Goal_message_typesupport_map = {
  2,
  "custom_interfaces",
  &_Maze_Goal_message_typesupport_ids.typesupport_identifier[0],
  &_Maze_Goal_message_typesupport_symbol_names.symbol_name[0],
  &_Maze_Goal_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t Maze_Goal_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_Maze_Goal_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_c

}  // namespace action

}  // namespace custom_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_C_EXPORT_custom_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, custom_interfaces, action, Maze_Goal)() {
  return &::custom_interfaces::action::rosidl_typesupport_c::Maze_Goal_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "custom_interfaces/msg/rosidl_typesupport_c__visibility_control.h"
// already included above
// #include "custom_interfaces/action/detail/maze__struct.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
// already included above
// #include "rosidl_typesupport_c/message_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_c/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace custom_interfaces
{

namespace action
{

namespace rosidl_typesupport_c
{

typedef struct _Maze_Result_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _Maze_Result_type_support_ids_t;

static const _Maze_Result_type_support_ids_t _Maze_Result_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _Maze_Result_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _Maze_Result_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _Maze_Result_type_support_symbol_names_t _Maze_Result_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, custom_interfaces, action, Maze_Result)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, custom_interfaces, action, Maze_Result)),
  }
};

typedef struct _Maze_Result_type_support_data_t
{
  void * data[2];
} _Maze_Result_type_support_data_t;

static _Maze_Result_type_support_data_t _Maze_Result_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _Maze_Result_message_typesupport_map = {
  2,
  "custom_interfaces",
  &_Maze_Result_message_typesupport_ids.typesupport_identifier[0],
  &_Maze_Result_message_typesupport_symbol_names.symbol_name[0],
  &_Maze_Result_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t Maze_Result_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_Maze_Result_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_c

}  // namespace action

}  // namespace custom_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_C_EXPORT_custom_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, custom_interfaces, action, Maze_Result)() {
  return &::custom_interfaces::action::rosidl_typesupport_c::Maze_Result_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "custom_interfaces/msg/rosidl_typesupport_c__visibility_control.h"
// already included above
// #include "custom_interfaces/action/detail/maze__struct.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
// already included above
// #include "rosidl_typesupport_c/message_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_c/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace custom_interfaces
{

namespace action
{

namespace rosidl_typesupport_c
{

typedef struct _Maze_Feedback_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _Maze_Feedback_type_support_ids_t;

static const _Maze_Feedback_type_support_ids_t _Maze_Feedback_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _Maze_Feedback_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _Maze_Feedback_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _Maze_Feedback_type_support_symbol_names_t _Maze_Feedback_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, custom_interfaces, action, Maze_Feedback)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, custom_interfaces, action, Maze_Feedback)),
  }
};

typedef struct _Maze_Feedback_type_support_data_t
{
  void * data[2];
} _Maze_Feedback_type_support_data_t;

static _Maze_Feedback_type_support_data_t _Maze_Feedback_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _Maze_Feedback_message_typesupport_map = {
  2,
  "custom_interfaces",
  &_Maze_Feedback_message_typesupport_ids.typesupport_identifier[0],
  &_Maze_Feedback_message_typesupport_symbol_names.symbol_name[0],
  &_Maze_Feedback_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t Maze_Feedback_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_Maze_Feedback_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_c

}  // namespace action

}  // namespace custom_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_C_EXPORT_custom_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, custom_interfaces, action, Maze_Feedback)() {
  return &::custom_interfaces::action::rosidl_typesupport_c::Maze_Feedback_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "custom_interfaces/msg/rosidl_typesupport_c__visibility_control.h"
// already included above
// #include "custom_interfaces/action/detail/maze__struct.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
// already included above
// #include "rosidl_typesupport_c/message_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_c/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace custom_interfaces
{

namespace action
{

namespace rosidl_typesupport_c
{

typedef struct _Maze_SendGoal_Request_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _Maze_SendGoal_Request_type_support_ids_t;

static const _Maze_SendGoal_Request_type_support_ids_t _Maze_SendGoal_Request_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _Maze_SendGoal_Request_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _Maze_SendGoal_Request_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _Maze_SendGoal_Request_type_support_symbol_names_t _Maze_SendGoal_Request_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, custom_interfaces, action, Maze_SendGoal_Request)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, custom_interfaces, action, Maze_SendGoal_Request)),
  }
};

typedef struct _Maze_SendGoal_Request_type_support_data_t
{
  void * data[2];
} _Maze_SendGoal_Request_type_support_data_t;

static _Maze_SendGoal_Request_type_support_data_t _Maze_SendGoal_Request_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _Maze_SendGoal_Request_message_typesupport_map = {
  2,
  "custom_interfaces",
  &_Maze_SendGoal_Request_message_typesupport_ids.typesupport_identifier[0],
  &_Maze_SendGoal_Request_message_typesupport_symbol_names.symbol_name[0],
  &_Maze_SendGoal_Request_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t Maze_SendGoal_Request_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_Maze_SendGoal_Request_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_c

}  // namespace action

}  // namespace custom_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_C_EXPORT_custom_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, custom_interfaces, action, Maze_SendGoal_Request)() {
  return &::custom_interfaces::action::rosidl_typesupport_c::Maze_SendGoal_Request_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "custom_interfaces/msg/rosidl_typesupport_c__visibility_control.h"
// already included above
// #include "custom_interfaces/action/detail/maze__struct.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
// already included above
// #include "rosidl_typesupport_c/message_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_c/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace custom_interfaces
{

namespace action
{

namespace rosidl_typesupport_c
{

typedef struct _Maze_SendGoal_Response_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _Maze_SendGoal_Response_type_support_ids_t;

static const _Maze_SendGoal_Response_type_support_ids_t _Maze_SendGoal_Response_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _Maze_SendGoal_Response_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _Maze_SendGoal_Response_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _Maze_SendGoal_Response_type_support_symbol_names_t _Maze_SendGoal_Response_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, custom_interfaces, action, Maze_SendGoal_Response)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, custom_interfaces, action, Maze_SendGoal_Response)),
  }
};

typedef struct _Maze_SendGoal_Response_type_support_data_t
{
  void * data[2];
} _Maze_SendGoal_Response_type_support_data_t;

static _Maze_SendGoal_Response_type_support_data_t _Maze_SendGoal_Response_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _Maze_SendGoal_Response_message_typesupport_map = {
  2,
  "custom_interfaces",
  &_Maze_SendGoal_Response_message_typesupport_ids.typesupport_identifier[0],
  &_Maze_SendGoal_Response_message_typesupport_symbol_names.symbol_name[0],
  &_Maze_SendGoal_Response_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t Maze_SendGoal_Response_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_Maze_SendGoal_Response_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_c

}  // namespace action

}  // namespace custom_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_C_EXPORT_custom_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, custom_interfaces, action, Maze_SendGoal_Response)() {
  return &::custom_interfaces::action::rosidl_typesupport_c::Maze_SendGoal_Response_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "custom_interfaces/msg/rosidl_typesupport_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
#include "rosidl_typesupport_c/service_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace custom_interfaces
{

namespace action
{

namespace rosidl_typesupport_c
{

typedef struct _Maze_SendGoal_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _Maze_SendGoal_type_support_ids_t;

static const _Maze_SendGoal_type_support_ids_t _Maze_SendGoal_service_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _Maze_SendGoal_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _Maze_SendGoal_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _Maze_SendGoal_type_support_symbol_names_t _Maze_SendGoal_service_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, custom_interfaces, action, Maze_SendGoal)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, custom_interfaces, action, Maze_SendGoal)),
  }
};

typedef struct _Maze_SendGoal_type_support_data_t
{
  void * data[2];
} _Maze_SendGoal_type_support_data_t;

static _Maze_SendGoal_type_support_data_t _Maze_SendGoal_service_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _Maze_SendGoal_service_typesupport_map = {
  2,
  "custom_interfaces",
  &_Maze_SendGoal_service_typesupport_ids.typesupport_identifier[0],
  &_Maze_SendGoal_service_typesupport_symbol_names.symbol_name[0],
  &_Maze_SendGoal_service_typesupport_data.data[0],
};

static const rosidl_service_type_support_t Maze_SendGoal_service_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_Maze_SendGoal_service_typesupport_map),
  rosidl_typesupport_c__get_service_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_c

}  // namespace action

}  // namespace custom_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_C_EXPORT_custom_interfaces
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_c, custom_interfaces, action, Maze_SendGoal)() {
  return &::custom_interfaces::action::rosidl_typesupport_c::Maze_SendGoal_service_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "custom_interfaces/msg/rosidl_typesupport_c__visibility_control.h"
// already included above
// #include "custom_interfaces/action/detail/maze__struct.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
// already included above
// #include "rosidl_typesupport_c/message_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_c/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace custom_interfaces
{

namespace action
{

namespace rosidl_typesupport_c
{

typedef struct _Maze_GetResult_Request_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _Maze_GetResult_Request_type_support_ids_t;

static const _Maze_GetResult_Request_type_support_ids_t _Maze_GetResult_Request_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _Maze_GetResult_Request_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _Maze_GetResult_Request_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _Maze_GetResult_Request_type_support_symbol_names_t _Maze_GetResult_Request_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, custom_interfaces, action, Maze_GetResult_Request)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, custom_interfaces, action, Maze_GetResult_Request)),
  }
};

typedef struct _Maze_GetResult_Request_type_support_data_t
{
  void * data[2];
} _Maze_GetResult_Request_type_support_data_t;

static _Maze_GetResult_Request_type_support_data_t _Maze_GetResult_Request_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _Maze_GetResult_Request_message_typesupport_map = {
  2,
  "custom_interfaces",
  &_Maze_GetResult_Request_message_typesupport_ids.typesupport_identifier[0],
  &_Maze_GetResult_Request_message_typesupport_symbol_names.symbol_name[0],
  &_Maze_GetResult_Request_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t Maze_GetResult_Request_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_Maze_GetResult_Request_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_c

}  // namespace action

}  // namespace custom_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_C_EXPORT_custom_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, custom_interfaces, action, Maze_GetResult_Request)() {
  return &::custom_interfaces::action::rosidl_typesupport_c::Maze_GetResult_Request_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "custom_interfaces/msg/rosidl_typesupport_c__visibility_control.h"
// already included above
// #include "custom_interfaces/action/detail/maze__struct.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
// already included above
// #include "rosidl_typesupport_c/message_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_c/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace custom_interfaces
{

namespace action
{

namespace rosidl_typesupport_c
{

typedef struct _Maze_GetResult_Response_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _Maze_GetResult_Response_type_support_ids_t;

static const _Maze_GetResult_Response_type_support_ids_t _Maze_GetResult_Response_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _Maze_GetResult_Response_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _Maze_GetResult_Response_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _Maze_GetResult_Response_type_support_symbol_names_t _Maze_GetResult_Response_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, custom_interfaces, action, Maze_GetResult_Response)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, custom_interfaces, action, Maze_GetResult_Response)),
  }
};

typedef struct _Maze_GetResult_Response_type_support_data_t
{
  void * data[2];
} _Maze_GetResult_Response_type_support_data_t;

static _Maze_GetResult_Response_type_support_data_t _Maze_GetResult_Response_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _Maze_GetResult_Response_message_typesupport_map = {
  2,
  "custom_interfaces",
  &_Maze_GetResult_Response_message_typesupport_ids.typesupport_identifier[0],
  &_Maze_GetResult_Response_message_typesupport_symbol_names.symbol_name[0],
  &_Maze_GetResult_Response_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t Maze_GetResult_Response_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_Maze_GetResult_Response_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_c

}  // namespace action

}  // namespace custom_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_C_EXPORT_custom_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, custom_interfaces, action, Maze_GetResult_Response)() {
  return &::custom_interfaces::action::rosidl_typesupport_c::Maze_GetResult_Response_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "custom_interfaces/msg/rosidl_typesupport_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
// already included above
// #include "rosidl_typesupport_c/service_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace custom_interfaces
{

namespace action
{

namespace rosidl_typesupport_c
{

typedef struct _Maze_GetResult_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _Maze_GetResult_type_support_ids_t;

static const _Maze_GetResult_type_support_ids_t _Maze_GetResult_service_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _Maze_GetResult_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _Maze_GetResult_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _Maze_GetResult_type_support_symbol_names_t _Maze_GetResult_service_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, custom_interfaces, action, Maze_GetResult)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, custom_interfaces, action, Maze_GetResult)),
  }
};

typedef struct _Maze_GetResult_type_support_data_t
{
  void * data[2];
} _Maze_GetResult_type_support_data_t;

static _Maze_GetResult_type_support_data_t _Maze_GetResult_service_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _Maze_GetResult_service_typesupport_map = {
  2,
  "custom_interfaces",
  &_Maze_GetResult_service_typesupport_ids.typesupport_identifier[0],
  &_Maze_GetResult_service_typesupport_symbol_names.symbol_name[0],
  &_Maze_GetResult_service_typesupport_data.data[0],
};

static const rosidl_service_type_support_t Maze_GetResult_service_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_Maze_GetResult_service_typesupport_map),
  rosidl_typesupport_c__get_service_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_c

}  // namespace action

}  // namespace custom_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_C_EXPORT_custom_interfaces
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_c, custom_interfaces, action, Maze_GetResult)() {
  return &::custom_interfaces::action::rosidl_typesupport_c::Maze_GetResult_service_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "custom_interfaces/msg/rosidl_typesupport_c__visibility_control.h"
// already included above
// #include "custom_interfaces/action/detail/maze__struct.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
// already included above
// #include "rosidl_typesupport_c/message_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_c/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace custom_interfaces
{

namespace action
{

namespace rosidl_typesupport_c
{

typedef struct _Maze_FeedbackMessage_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _Maze_FeedbackMessage_type_support_ids_t;

static const _Maze_FeedbackMessage_type_support_ids_t _Maze_FeedbackMessage_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _Maze_FeedbackMessage_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _Maze_FeedbackMessage_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _Maze_FeedbackMessage_type_support_symbol_names_t _Maze_FeedbackMessage_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, custom_interfaces, action, Maze_FeedbackMessage)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, custom_interfaces, action, Maze_FeedbackMessage)),
  }
};

typedef struct _Maze_FeedbackMessage_type_support_data_t
{
  void * data[2];
} _Maze_FeedbackMessage_type_support_data_t;

static _Maze_FeedbackMessage_type_support_data_t _Maze_FeedbackMessage_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _Maze_FeedbackMessage_message_typesupport_map = {
  2,
  "custom_interfaces",
  &_Maze_FeedbackMessage_message_typesupport_ids.typesupport_identifier[0],
  &_Maze_FeedbackMessage_message_typesupport_symbol_names.symbol_name[0],
  &_Maze_FeedbackMessage_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t Maze_FeedbackMessage_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_Maze_FeedbackMessage_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_c

}  // namespace action

}  // namespace custom_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_C_EXPORT_custom_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, custom_interfaces, action, Maze_FeedbackMessage)() {
  return &::custom_interfaces::action::rosidl_typesupport_c::Maze_FeedbackMessage_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

#include "action_msgs/msg/goal_status_array.h"
#include "action_msgs/srv/cancel_goal.h"
#include "custom_interfaces/action/maze.h"
#include "custom_interfaces/action/detail/maze__type_support.h"

static rosidl_action_type_support_t _custom_interfaces__action__Maze__typesupport_c;

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_C_EXPORT_custom_interfaces
const rosidl_action_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__ACTION_SYMBOL_NAME(
  rosidl_typesupport_c, custom_interfaces, action, Maze)()
{
  // Thread-safe by always writing the same values to the static struct
  _custom_interfaces__action__Maze__typesupport_c.goal_service_type_support =
    ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(
    rosidl_typesupport_c, custom_interfaces, action, Maze_SendGoal)();
  _custom_interfaces__action__Maze__typesupport_c.result_service_type_support =
    ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(
    rosidl_typesupport_c, custom_interfaces, action, Maze_GetResult)();
  _custom_interfaces__action__Maze__typesupport_c.cancel_service_type_support =
    ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(
    rosidl_typesupport_c, action_msgs, srv, CancelGoal)();
  _custom_interfaces__action__Maze__typesupport_c.feedback_message_type_support =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
    rosidl_typesupport_c, custom_interfaces, action, Maze_FeedbackMessage)();
  _custom_interfaces__action__Maze__typesupport_c.status_message_type_support =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
    rosidl_typesupport_c, action_msgs, msg, GoalStatusArray)();

  return &_custom_interfaces__action__Maze__typesupport_c;
}

#ifdef __cplusplus
}
#endif
