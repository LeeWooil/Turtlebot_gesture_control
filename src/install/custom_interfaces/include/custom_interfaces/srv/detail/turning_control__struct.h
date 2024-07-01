// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from custom_interfaces:srv/TurningControl.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_INTERFACES__SRV__DETAIL__TURNING_CONTROL__STRUCT_H_
#define CUSTOM_INTERFACES__SRV__DETAIL__TURNING_CONTROL__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Struct defined in srv/TurningControl in the package custom_interfaces.
typedef struct custom_interfaces__srv__TurningControl_Request
{
  uint32_t time_duration;
  double angular_vel_z;
  double linear_vel_x;
} custom_interfaces__srv__TurningControl_Request;

// Struct for a sequence of custom_interfaces__srv__TurningControl_Request.
typedef struct custom_interfaces__srv__TurningControl_Request__Sequence
{
  custom_interfaces__srv__TurningControl_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} custom_interfaces__srv__TurningControl_Request__Sequence;


// Constants defined in the message

// Struct defined in srv/TurningControl in the package custom_interfaces.
typedef struct custom_interfaces__srv__TurningControl_Response
{
  bool success;
} custom_interfaces__srv__TurningControl_Response;

// Struct for a sequence of custom_interfaces__srv__TurningControl_Response.
typedef struct custom_interfaces__srv__TurningControl_Response__Sequence
{
  custom_interfaces__srv__TurningControl_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} custom_interfaces__srv__TurningControl_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CUSTOM_INTERFACES__SRV__DETAIL__TURNING_CONTROL__STRUCT_H_
