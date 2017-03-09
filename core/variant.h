#pragma once

#include "debug.h"
#include "traits.h"
#include <stdexcept>
#include <utility>

namespace core {
namespace impl {

template <typename Index, Index, typename...> struct variant {
  template <typename>[[noreturn]] static Index index_of() { ASSERT_UNREACHABLE(); }

  template <typename U>[[noreturn]] static void construct(Index, void*, U&&) {
    ASSERT_UNREACHABLE();
  }

  template <typename U>[[noreturn]] static void destruct(Index, U&&) { ASSERT_UNREACHABLE(); }

  template <typename L, typename R>[[noreturn]] static void assign_same(Index, L&&, R&&) {
    ASSERT_UNREACHABLE();
  }

  template <typename L, typename R>[[noreturn]] static void swap_same(Index, L&&, R&&) {
    ASSERT_UNREACHABLE();
  }

  template <typename Result, typename Fn, typename U, typename... Args>
  [[noreturn]] static Result apply(Index, Fn&&, U&&, Args&&...) {
    ASSERT_UNREACHABLE();
  }
};

template <typename Index, Index N, typename T, typename... Ts> struct variant<Index, N, T, Ts...> {
  using tail = variant<Index, N + 1, Ts...>;

  static constexpr Index index() { return N; }

  template <typename Q> static constexpr Index index_of() {
    return std::is_same<Q, T>::value ? index() : tail::template index_of<Q>();
  }

  template <typename U>
  using fwd_type = typename std::conditional<
      std::is_rvalue_reference<U&&>::value,
      typename std::conditional<std::is_const<typename std::remove_reference<U&&>::type>::value,
                                const T&&, T&&>::type,
      typename std::conditional<std::is_const<typename std::remove_reference<U&&>::type>::value,
                                const T&, T&>::type>::type;

  template <typename U> static auto fwd(U&& data) -> fwd_type<U> {
    return reinterpret_cast<fwd_type<U>>(data);
  }

  template <typename U> static void construct(Index i, void* storage, U&& data) {
    if (i == index())
      new (storage) T(fwd<U>(std::forward<U>(data)));
    else
      tail::construct(i, storage, std::forward<U>(data));
  }

  template <typename U> static void destruct(Index i, U&& data) {
    if (i == index())
      fwd<U>(std::forward<U>(data)).~T();
    else
      tail::destruct(i, std::forward<U>(data));
  }

  template <typename L, typename R> static void assign_same(Index i, L&& lhs, R&& rhs) {
    if (i == index())
      fwd<L>(std::forward<L>(lhs)) = fwd<R>(std::forward<R>(rhs));
    else
      tail::assign_same(i, std::forward<L>(lhs), std::forward<R>(rhs));
  }

  template <typename L, typename R> static void swap_same(Index i, L&& lhs, R&& rhs) {
    using std::swap;
    if (i == index())
      swap(fwd<L>(std::forward<L>(lhs)), fwd<R>(std::forward<R>(rhs)));
    else
      tail::swap_same(i, std::forward<L>(lhs), std::forward<R>(rhs));
  }

  template <typename Result, typename Fn, typename U, typename... Args>
  static Result apply(Index i, Fn&& fn, U&& data, Args&&... args) {
    if (i == index())
      return fn(fwd<U>(std::forward<U>(data)), std::forward<Args>(args)...);
    return tail::template apply<Result>(i, std::forward<Fn>(fn), std::forward<U>(data),
                                        std::forward<Args>(args)...);
  }
};

} // namespace impl

/**
 * @brief bad_variant_access
 */

struct bad_variant_access : std::exception {};

/**
 * @brief variant<typename...>
 */

template <typename... Types> struct variant {
  static_assert(all_unique<Types...>::value, "every enum`s type must be unique");

  using index_type = unsigned;
  using value_type = typename std::aligned_union<1, Types...>::type;

public:
  template <typename T = typename fst<Types...>::type>
  variant()
      : _index(index_of<T>()) {
    new (&_data) T();
  }

  template <typename T, typename U = typename std::decay<T>::type,
            typename = typename std::enable_if<contains<U, Types...>::value>::type>
  variant(T&& value)
      : _index(index_of<U>()) {
    new (&_data) U(std::forward<T>(value));
  }

  variant(variant&& other)
      : _index(other._index) {
    _impl::construct(index(), &_data, std::move(other._data));
  }

  variant(const variant& other)
      : _index(other._index) {
    _impl::construct(index(), &_data, other._data);
  }

  ~variant() { _impl::destruct(index(), _data); }

public:
  template <typename T, typename U = typename std::decay<T>::type,
            typename = typename std::enable_if<contains<U, Types...>::value>::type>
  variant& operator=(T&& value) {
    if (is<U>())
      as<U>() = std::forward<T>(value);
    else
      emplace<U>(std::forward<T>(value));
    return *this;
  }

  variant& operator=(variant&& other) {
    if (index() == other.index())
      _impl::assign_same(index(), _data, std::move(other._data));
    else {
      _impl::destruct(index(), _data);
      _impl::construct(other.index(), _data, std::move(other._data));
      _index = other.index();
    }
    return *this;
  }

  variant& operator=(const variant& other) {
    if (index() == other.index())
      _impl::assign_same(index(), _data, other._data);
    else {
      _impl::destruct(index(), _data);
      _impl::construct(other.index(), _data, other._data);
      _index = other.index();
    }
    return *this;
  }

  void swap(variant& other) {
    if (index() == other.index())
      _impl::swap_same(index(), _data, other._data);
    else {
      variant tmp(*this);
      (*this) == other;
      other = tmp;
    }
  }

  friend void swap(variant& lhs, variant& rhs) { lhs.swap(rhs); }

public:
  template <typename U, typename... Args,
            typename = typename std::enable_if<contains<U, Types...>::value>::type>
  void emplace(Args&&... args) {
    _impl::destruct(index(), _data);

    new (&_data) U(std::forward<Args>(args)...);
    _index = index_of<U>();
  }

public:
  index_type index() const { return _index; }

  template <typename Q> bool is() const { return index() == index_of<Q>(); }

  template <typename Q> static constexpr index_type index_of() {
    static_assert(contains<Q, Types...>::value, "enum does not handles queried type");
    return _impl::template index_of<Q>();
  }

public:
  template <typename Fn, typename... Args,
            typename Result = typename std::result_of<Fn(typename fst<Types...>::type)>::type>
  Result apply(Fn&& fn, Args&&... args) & {
    static_assert(all_same<typename std::result_of<Fn(Types)>::type...>::value,
                  "all overloaded variants must have the same return type");
    return _impl::template apply<Result>(index(), std::forward<Fn>(fn), _data,
                                         std::forward<Args>(args)...);
  }

  template <typename Fn, typename... Args,
            typename Result = typename std::result_of<Fn(typename fst<Types...>::type)>::type>
  Result apply(Fn&& fn, Args&&... args) && {
    static_assert(all_same<typename std::result_of<Fn(Types)>::type...>::value,
                  "all overloaded variants must have the same return type");
    return _impl::template apply<Result>(index(), std::forward<Fn>(fn), std::move(_data),
                                         std::forward<Args>(args)...);
  }

  template <typename Fn, typename... Args,
            typename Result = typename std::result_of<Fn(typename fst<Types...>::type)>::type>
  Result apply(Fn&& fn, Args&&... args) const& {
    static_assert(all_same<typename std::result_of<Fn(Types)>::type...>::value,
                  "all overloaded variants must have the same return type");
    return _impl::template apply<Result>(index(), std::forward<Fn>(fn), _data,
                                         std::forward<Args>(args)...);
  }

public:
  template <typename U> U& as() & {
    if (index() != index_of<U>())
      throw bad_variant_access{};
    return reinterpret_cast<U&>(_data);
  }

  template <typename U> U&& as() && {
    if (index() != index_of<U>())
      throw bad_variant_access{};
    return reinterpret_cast<U&&>(_data);
  }

  template <typename U> const U& as() const& {
    if (index() != index_of<U>())
      throw bad_variant_access{};
    return reinterpret_cast<const U&>(_data);
  }

private:
  index_type _index;
  value_type _data;

private:
  using _impl = impl::variant<index_type, 0, Types...>;
};

} // namespace core
