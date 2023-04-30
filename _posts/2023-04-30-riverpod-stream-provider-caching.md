---
layout: post
title: Riverpod Stream Provider Caching
description: Make Streams Great Again
image: /assets/images/riverpod-stream-provider-cache/riverpod.png
project: false
permalink: "/blog/:title/"
source: https://github.com/Blacksuan19/stream_provider_cache
tags:
  - flutter
  - riverpod
---

## Introduction

Riverpod is a popular state management solution for Flutter applications that
provides various providers to manage state. One of these providers is
StreamProvider, which is used to manage state that comes from a stream, such as
data from an API or database. Additionally, Riverpod also provides provider
families, which allow you to create multiple instances of the same provider with
different configurations. In this post, we will explore how to use
StreamProvider and provider families in Riverpod, in addition to how to
implement caching for StreamProvider to improve the performance of our
application.

In Riverpod, provider families are a powerful tool that allows us to create
multiple instances of a provider with different arguments. One type of provider
family is the `StreamProvider`, which is particularly useful when we need to
fetch data asynchronously and stream it to our widgets. For example, suppose we
want to fetch a list of articles from an API with different categories. In this
case, we can create a `newsProviderFamily` that takes in a `category` argument
and returns a `StreamProvider` that fetches the articles for that category. By
using this provider family, we can easily display articles for different
categories in our app by simply calling the `newsProviderFamily` with the
appropriate category argument. The `StreamProvider` will handle fetching the
data and streaming it to our widgets, while also allowing us to implement
caching and error handling. Overall, provider families with `StreamProvider` are
a powerful combination that can simplify our code and make it more efficient.

## Provider Families

Provider families can accept any type of argument as an input, however, to use a
custom args model as a ProviderFamily argument, the model needs to be compreable
to an instance of itself by either extending `Equitable` class from
[equitable](https://pub.dev/packages/equatable) or manually implementing
`bool operator ==(Object other)` and `int get hashCode` methods.

```dart

// data model
class GridItemModel extends Jsonable {
  final int index;
  final int code;

  GridItemModel({
    required this.index,
    required this.code,
  });
}

// riverpod provider family arguments
class GridItemsProviderArgs {
  final SharedPrefKey prefsKey;
  final SharedPrefKey refreshKey;
  final StringProvider filterProvider

  GridItemsProviderArgs({
    required this.prefsKey,
    required this.refreshKey,
    required this.filterProvider,
  });

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;

    return other is GridItemsProviderArgs &&
        other.prefsKey == prefsKey &&
        other.refreshKey == refreshKey &&
        other.filterProvider == filterProvider;
  }

  @override
  int get hashCode => prefsKey.hashCode ^ refreshKey.hashCode ^ filterProvider.hashCode;
}
```

the above models allows us to create multiple providers from the same code with
different arguments, we can create a `StreamProvider` family for demonstration
as follows

```dart
final gridItemsProviderFamily = StreamProvider.autoDispose
    .family<List<GridItemModel>, GridItemsProviderArgs>((ref, args) async* {
  var allItems = <GridItemModel>[];
  for (int i = allItems.length; i < 100; i++) {
    // small delay to simulate network latency
    await Future.delayed(const Duration(milliseconds: 500));

    // generate a unique key for each item
    final uniqueKey = UniqueKey().hashCode;

    // add new item to the stream
    allItems = [...allItems, GridItemModel(index: i, code: uniqueKey)];

    yield allItems;
  }
});
```

we can use the above provider as we would any other with the exception of
passing an instance of `GridItemsProviderArgs` to the provider as follows

```dart
final allItems = ref.watch(
  gridItemsProviderFamily(
	GridItemsProviderArgs(
	  prefsKey: SharedPrefKey.providerCache,
	  refreshKey: SharedPrefKey.shouldRefreshProviderCache,
	),
  ),
),
```

## The Problem

[By default](https://github.com/rrousselGit/riverpod/issues/1344),
StreamProvider does not support pausing the stream. This means that there is no
cache of the stream itself and it will have to be fetched every time the
provider is used in any widget, even if the data has not changed, this coupled
with `autoDispose` will force the app to re-fetch the stream every time a page
that uses it is loaded (even during the same app session), not using
`autoDispose` would not be ideal since it means the provider will start loading
when the user navigates to a page that contains the provider and will continue
to load until the stream is finished, that could be fine when the stream is
small but it would be a performance nightmare if the stream is large or creating
an `Item` object takes a while leading to many unnecessary network requests and
slower performance. the gif below shows the problem in action, the stream is
fetched every time the page is loaded even if the data has not changed.

<p style="text-align:center;">
<image
    src="/assets/images/riverpod-stream-provider-cache/no_cache.gif"
    alt="without cache"
    height="500"></image>
</p>

## The Solution

To address the pausing issue, we can implement caching in our StreamProvider
using shared_preferences. Shared_preferences is a package that provides a simple
key-value store for persisting data on the device. We can use it to store the
data fetched from the stream and retrieve it when the widget is rebuilt. This
way, we can avoid making unnecessary network requests and improve the
performance of our app. we can boil it down to below steps

- Inside the provider, we first try to get the data from cache.
- If the data is found, we resume the stream from the cached items.
- Otherwise, we start with an empty list.
- Use any of the data model fields to check whether it is present in the cache.
- if the item is present, yield the list of cached items without adding the item
  again.
- if the item is not present, add the item to the last of items.
- cache entire list of items.
- perform any processing needed for the list (filter, order, reduce...etc).
- yield list of cached items

```dart
final gridItemsProviderFamily = StreamProvider.autoDispose
    .family<List<GridItemModel>, GridItemsProviderArgs>((ref, args) async* {
  // fetch data from cache if present
  List<GridItemModel>? cachedItems = await getCachedItems(
    prefsKey: args.prefsKey,
    refreshKey: args.refreshKey,
  );

  // resume the stream from the cached items
  var allItems = cachedItems ?? <GridItemModel>[];
  for (int i = allItems.length; i < allItems.length + 100; i++) {
    // small delay to simulate network latency
    await Future.delayed(const Duration(milliseconds: 500));

    // generate a unique key for each item
    final uniqueKey = UniqueKey().hashCode;

    // check if the item is already in the cache
    // this is an extra check to avoid duplicates if index is off
    if (allItems.any((element) => element.code == uniqueKey)) {
      yield allItems;
      continue;
    }

    // add new items to the stream
    allItems = [...allItems, GridItemModel(index: i, code: uniqueKey)];

    // cache items to shared preferences
    await cacheItems(key: args.prefsKey, items: allItems);

    // filter items based on the filter provider
    allItems = filterItems(
      ref: ref,
      filterProvider: args.filterProvider,
      items: allItems,
    );

    yield allItems;
  }
});

```

With the above provider code we solve all our previous issues:

- we can now pause and resume a stream whenever we want without losing progress.
- `autoDispose` works as it would in other cases.
- we can trigger a provider refresh remotely (by using Firebase Remote Config or
  similar).

if we print the length of `allItems` we will see that it resumes from where the
cache stops every time the provider is requested unless the refresh key value is
true in the store.

<p style="text-align:center;">
<image
    src="/assets/images/riverpod-stream-provider-cache/cached.gif"
    alt="without cache"
    height="500"></image>
</p>

we can see that the stream is resumed from the cache in the above gif, click the
source button on the left to see the full demo application code.

Riverpod provides several state management solutions for Flutter applications,
including StateProvider, StreamProvider, and Provider families. With Riverpod,
you can easily manage the state of your application and build performant and
maintainable applications. Overall, implementing caching inside a StreamProvider
can greatly improve the performance of our app by reducing the number of network
requests and improving the user experience.
