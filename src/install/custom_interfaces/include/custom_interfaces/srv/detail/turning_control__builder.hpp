// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from custom_interfaces:srv/TurningControl.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_INTERFACES__SRV__DETAIL__TURNING_CONTROL__BUILDER_HPP_
#define CUSTOM_INTERFACES__SRV__DETAIL__TURNING_CONTROL__BUILDER_HPP_

#include "custom_interfaces/srv/detail/turning_control__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace custom_interfaces
{

namespace srv
{

namespace builder
{

class Init_TurningControl_Request_linear_vel_x
{
public:
  explicit Init_TurningControl_Request_linear_vel_x(::custom_interfaces::srv::TurningControl_Request & msg)
  : msg_(msg)
  {}
  ::custom_interfaces::srv::TurningControl_Request linear_vel_x(::custom_interfaces::srv::TurningControl_Request::_linear_vel_x_type arg)
  {
    msg_.linear_vel_x = std::move(arg);
    return std::move(msg_);
  }

private:
  ::custom_interfaces::srv::TurningControl_Request msg_;
};

class Init_TurningControl_Request_angular_vel_z
{
public:
  explicit Init_TurningControl_Request_angular_vel_z(::custom_interfaces::srv::TurningControl_Request & msg)
  : msg_(msg)
  {}
  Init_TurningControl_Request_linear_vel_x angular_vel_z(::custom_interfaces::srv::TurningControl_Request::_angular_vel_z_type arg)
  {
    msg_.angular_vel_z = std::move(arg);
    return Init_TurningControl_Request_linear_vel_x(msg_);
  }

private:
  ::custom_interfaces::srv::TurningControl_Request msg_;
};

class Init_TurningControl_Request_time_duration
{
public:
  Init_TurningControl_Request_time_duration()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_TurningControl_Request_angular_vel_z time_duration(::custom_interfaces::srv::TurningControl_Request::_time_duration_type arg)
  {
    msg_.time_duration = std::move(arg);
    return Init_TurningControl_Request_angular_vel_z(msg_);
  }

private:
  ::custom_interfaces::srv::TurningControl_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::custom_interfaces::srv::TurningControl_Request>()
{
  return custom_interfaces::srv::builder::Init_TurningControl_Request_time_duration();
}

}  // namespace custom_interfaces


namespace custom_interfaces
{

namespace srv
{

namespace builder
{

class Init_TurningControl_Response_success
{
public:
  Init_TurningControl_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::custom_interfaces::srv::TurningControl_Response success(::custom_interfaces::srv::TurningControl_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::custom_interfaces::srv::TurningControl_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::custom_interfaces::srv::TurningControl_Response>()
{
  return custom_interfaces::srv::builder::Init_TurningControl_Response_success();
}

}  // namespace custom_interfaces

#endif  // CUSTOM_INTERFACES__SRV__DETAIL__TURNING_CONTROL__BUILDER_HPP_
