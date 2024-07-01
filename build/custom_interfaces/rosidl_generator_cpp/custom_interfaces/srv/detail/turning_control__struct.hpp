// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from custom_interfaces:srv/TurningControl.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_INTERFACES__SRV__DETAIL__TURNING_CONTROL__STRUCT_HPP_
#define CUSTOM_INTERFACES__SRV__DETAIL__TURNING_CONTROL__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


#ifndef _WIN32
# define DEPRECATED__custom_interfaces__srv__TurningControl_Request __attribute__((deprecated))
#else
# define DEPRECATED__custom_interfaces__srv__TurningControl_Request __declspec(deprecated)
#endif

namespace custom_interfaces
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct TurningControl_Request_
{
  using Type = TurningControl_Request_<ContainerAllocator>;

  explicit TurningControl_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->time_duration = 0ul;
      this->angular_vel_z = 0.0;
      this->linear_vel_x = 0.0;
    }
  }

  explicit TurningControl_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->time_duration = 0ul;
      this->angular_vel_z = 0.0;
      this->linear_vel_x = 0.0;
    }
  }

  // field types and members
  using _time_duration_type =
    uint32_t;
  _time_duration_type time_duration;
  using _angular_vel_z_type =
    double;
  _angular_vel_z_type angular_vel_z;
  using _linear_vel_x_type =
    double;
  _linear_vel_x_type linear_vel_x;

  // setters for named parameter idiom
  Type & set__time_duration(
    const uint32_t & _arg)
  {
    this->time_duration = _arg;
    return *this;
  }
  Type & set__angular_vel_z(
    const double & _arg)
  {
    this->angular_vel_z = _arg;
    return *this;
  }
  Type & set__linear_vel_x(
    const double & _arg)
  {
    this->linear_vel_x = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    custom_interfaces::srv::TurningControl_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const custom_interfaces::srv::TurningControl_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<custom_interfaces::srv::TurningControl_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<custom_interfaces::srv::TurningControl_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      custom_interfaces::srv::TurningControl_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<custom_interfaces::srv::TurningControl_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      custom_interfaces::srv::TurningControl_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<custom_interfaces::srv::TurningControl_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<custom_interfaces::srv::TurningControl_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<custom_interfaces::srv::TurningControl_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__custom_interfaces__srv__TurningControl_Request
    std::shared_ptr<custom_interfaces::srv::TurningControl_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__custom_interfaces__srv__TurningControl_Request
    std::shared_ptr<custom_interfaces::srv::TurningControl_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const TurningControl_Request_ & other) const
  {
    if (this->time_duration != other.time_duration) {
      return false;
    }
    if (this->angular_vel_z != other.angular_vel_z) {
      return false;
    }
    if (this->linear_vel_x != other.linear_vel_x) {
      return false;
    }
    return true;
  }
  bool operator!=(const TurningControl_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct TurningControl_Request_

// alias to use template instance with default allocator
using TurningControl_Request =
  custom_interfaces::srv::TurningControl_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace custom_interfaces


#ifndef _WIN32
# define DEPRECATED__custom_interfaces__srv__TurningControl_Response __attribute__((deprecated))
#else
# define DEPRECATED__custom_interfaces__srv__TurningControl_Response __declspec(deprecated)
#endif

namespace custom_interfaces
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct TurningControl_Response_
{
  using Type = TurningControl_Response_<ContainerAllocator>;

  explicit TurningControl_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  explicit TurningControl_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  // field types and members
  using _success_type =
    bool;
  _success_type success;

  // setters for named parameter idiom
  Type & set__success(
    const bool & _arg)
  {
    this->success = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    custom_interfaces::srv::TurningControl_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const custom_interfaces::srv::TurningControl_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<custom_interfaces::srv::TurningControl_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<custom_interfaces::srv::TurningControl_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      custom_interfaces::srv::TurningControl_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<custom_interfaces::srv::TurningControl_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      custom_interfaces::srv::TurningControl_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<custom_interfaces::srv::TurningControl_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<custom_interfaces::srv::TurningControl_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<custom_interfaces::srv::TurningControl_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__custom_interfaces__srv__TurningControl_Response
    std::shared_ptr<custom_interfaces::srv::TurningControl_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__custom_interfaces__srv__TurningControl_Response
    std::shared_ptr<custom_interfaces::srv::TurningControl_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const TurningControl_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const TurningControl_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct TurningControl_Response_

// alias to use template instance with default allocator
using TurningControl_Response =
  custom_interfaces::srv::TurningControl_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace custom_interfaces

namespace custom_interfaces
{

namespace srv
{

struct TurningControl
{
  using Request = custom_interfaces::srv::TurningControl_Request;
  using Response = custom_interfaces::srv::TurningControl_Response;
};

}  // namespace srv

}  // namespace custom_interfaces

#endif  // CUSTOM_INTERFACES__SRV__DETAIL__TURNING_CONTROL__STRUCT_HPP_
