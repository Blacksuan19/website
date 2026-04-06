/*
  Forty by HTML5 UP
  html5up.net | @ajlkn
  Free for personal and commercial use under the CCA 3.0 license (html5up.net/license)
*/

(function ($) {
  skel.breakpoints({
    xlarge: "(max-width: 1680px)",
    large: "(max-width: 1280px)",
    medium: "(max-width: 980px)",
    small: "(max-width: 736px)",
    xsmall: "(max-width: 480px)",
    xxsmall: "(max-width: 360px)",
  });

  /**
   * Applies parallax scrolling to an element's background image.
   * @return {jQuery} jQuery object.
   */
  $.fn._parallax =
    skel.vars.browser == "ie" || skel.vars.browser == "edge" || skel.vars.mobile
      ? function () {
        return $(this);
      }
      : function (intensity) {
        var $window = $(window),
          $this = $(this);

        if (this.length == 0 || intensity === 0) return $this;

        if (this.length > 1) {
          for (var i = 0; i < this.length; i++)
            $(this[i])._parallax(intensity);

          return $this;
        }

        if (!intensity) intensity = 0.25;

        $this.each(function () {
          var $t = $(this),
            on,
            off;

          on = function () {
            $t.css(
              "background-position",
              "center 100%, center 100%, center 0px"
            );

            $window.on("scroll._parallax", function () {
              var pos =
                parseInt($window.scrollTop()) - parseInt($t.position().top);

              $t.css(
                "background-position",
                "center " + pos * (-1 * intensity) + "px"
              );
            });
          };

          off = function () {
            $t.css("background-position", "");

            $window.off("scroll._parallax");
          };

          skel.on("change", function () {
            if (skel.breakpoint("medium").active) off();
            else on();
          });
        });

        $window
          .off("load._parallax resize._parallax")
          .on("load._parallax resize._parallax", function () {
            $window.trigger("scroll");
          });

        return $(this);
      };

  $(function () {
    var $window = $(window),
      $body = $("body"),
      $wrapper = $("#wrapper"),
      $header = $("#header"),
      $banner = $("#banner"),
      $navDropdowns = $("#header .nav-dropdown"),
      $navDropdownTriggers = $("#header .nav-dropdown-trigger"),
      $searchToggle = $('[data-search-toggle="true"]'),
      $searchForm = $("#site-search-form"),
      $searchInput = $("#site-search-input");

    function initPostToc() {
      var $toc = $("[data-post-toc]").first();
      var $content = $("[data-post-content]").first();
      var $nav = $toc.find("[data-post-toc-nav]");
      var $toggle = $toc.find("[data-post-toc-toggle]");
      var $toggleText = $toc.find("[data-post-toc-toggle-text]");
      var $postTags = $(".post-tags").first();
      var $postHeader = $(".post-header").first();
      var $scrollButton = $("#scroll");
      var desktopRailMediaQuery = window.matchMedia("(min-width: 737px)");
      var wideRailMediaQuery = window.matchMedia("(min-width: 1481px)");
      var desktopCollapsed = false;

      if ($toc.length === 0 || $content.length === 0 || $nav.length === 0) {
        return;
      }

      var $links = $nav.find("a[href^='#']");
      var headings = $content.find("h2[id], h3[id]").toArray();
      var linkMap = {};

      if ($links.length === 0 || headings.length === 0) {
        return;
      }

      $links.each(function () {
        var $link = $(this);
        var href = $link.attr("href") || "";
        var headingId = href.slice(1);

        if (headingId) {
          linkMap[headingId] = $link;
        }
      });

      function isDesktopRail() {
        return desktopRailMediaQuery.matches;
      }

      function prefersCollapsedRail() {
        return isDesktopRail() && !wideRailMediaQuery.matches;
      }

      function resetDesktopCollapseState() {
        desktopCollapsed = prefersCollapsedRail();
      }

      function getDocumentTop($element) {
        if (!$element.length) {
          return null;
        }

        return window.scrollY + $element[0].getBoundingClientRect().top;
      }

      function updateDesktopRailMetrics() {
        if (!isDesktopRail()) {
          $toc[0].style.removeProperty("--post-toc-chip-top");
          $toc[0].style.removeProperty("--post-toc-panel-top");
          $toc[0].style.removeProperty("--post-toc-panel-max-height");
          return;
        }

        var anchorTop = getDocumentTop($postTags);

        if (anchorTop === null) {
          anchorTop = getDocumentTop($postHeader);
        }

        if (anchorTop === null) {
          return;
        }

        var toggleHeight = $toggle.outerHeight() || 0;
        var tagsHeight = $postTags.length ? $postTags.outerHeight() || 0 : 0;
        var chipTop = Math.max(anchorTop + Math.max((tagsHeight - toggleHeight) / 2, 0), $header.outerHeight() + 16);
        var panelTop = chipTop + toggleHeight + 12;
        var scrollReserved = ($scrollButton.length ? $scrollButton.outerHeight() || 0 : 0) + 40;
        var panelMaxHeight = Math.max(window.innerHeight - panelTop - scrollReserved, 160);

        $toc[0].style.setProperty("--post-toc-chip-top", chipTop + "px");
        $toc[0].style.setProperty("--post-toc-panel-top", panelTop + "px");
        $toc[0].style.setProperty("--post-toc-panel-max-height", panelMaxHeight + "px");
      }

      function syncTocState() {
        var expanded = isDesktopRail() ? !desktopCollapsed : $toc.hasClass("is-expanded");

        $toc.toggleClass("is-desktop-rail", isDesktopRail());
        $toc.toggleClass("is-collapsed", !expanded);

        if (expanded) {
          $toc.addClass("is-expanded");
        } else {
          $toc.removeClass("is-expanded");
        }

        $toggle.attr("aria-expanded", expanded ? "true" : "false");

        if (isDesktopRail()) {
          $toggleText.text(expanded ? "Hide sections" : "On this page");
        } else {
          $toggleText.text(expanded ? "Hide sections" : "Show sections");
        }

        updateDesktopRailMetrics();
      }

      function updateActiveHeading() {
        var scrollPosition = window.scrollY + $header.outerHeight() + 32;
        var activeId = headings[0].id;

        headings.forEach(function (heading) {
          if (heading.offsetTop <= scrollPosition) {
            activeId = heading.id;
          }
        });

        $.each(linkMap, function (headingId, $link) {
          $link.toggleClass("is-active", headingId === activeId);
        });
      }

      function scrollToHeading(headingId) {
        var target = document.getElementById(headingId);

        if (!target) {
          return;
        }

        var scrollTop = window.scrollY + target.getBoundingClientRect().top - $header.outerHeight() - 18;

        window.history.replaceState(null, "", "#" + headingId);
        window.scrollTo({
          top: Math.max(scrollTop, 0),
          behavior: "smooth",
        });
      }

      $toggle.on("click", function () {
        if (isDesktopRail()) {
          desktopCollapsed = !desktopCollapsed;
        } else {
          $toc.toggleClass("is-expanded");
        }

        syncTocState();
      });

      $nav.on("click", "a", function (event) {
        var href = $(this).attr("href") || "";
        var headingId = href.slice(1);

        if (!headingId) {
          return;
        }

        event.preventDefault();
        scrollToHeading(headingId);

        syncTocState();
      });

      function handleRailBreakpointChange() {
        if (!isDesktopRail()) {
          $toc.removeClass("is-expanded");
        }

        resetDesktopCollapseState();
        syncTocState();
        updateActiveHeading();
      }

      if (desktopRailMediaQuery.addEventListener) {
        desktopRailMediaQuery.addEventListener("change", handleRailBreakpointChange);
        wideRailMediaQuery.addEventListener("change", handleRailBreakpointChange);
      } else {
        desktopRailMediaQuery.addListener(handleRailBreakpointChange);
        wideRailMediaQuery.addListener(handleRailBreakpointChange);
      }

      resetDesktopCollapseState();

      if (!isDesktopRail()) {
        $toc.removeClass("is-expanded");
      }

      syncTocState();
      updateDesktopRailMetrics();
      updateActiveHeading();
      $window.on("scroll", updateActiveHeading);
      $window.on("resize", function () {
        updateDesktopRailMetrics();
        updateActiveHeading();
      });
    }

    // Clear transitioning state on unload/hide.
    $window.on("unload pagehide", function () {
      window.setTimeout(function () {
        $(".is-transitioning").removeClass("is-transitioning");
      }, 250);
    });

    // Fix: Enable IE-only tweaks.
    if (skel.vars.browser == "ie" || skel.vars.browser == "edge")
      $body.addClass("is-ie");

    // Fix: Placeholder polyfill.
    $("form").placeholder();

    // Prioritize "important" elements on medium.
    skel.on("+medium -medium", function () {
      $.prioritize(
        ".important\\28 medium\\29",
        skel.breakpoint("medium").active
      );
    });

    // Scrolly.
    $(".scrolly").scrolly({
      offset: function () {
        return $header.height() - 2;
      },
    });

    // Tiles.
    var $tiles = $(".tiles > article");

    initPostToc();

    $tiles.each(function () {
      var $this = $(this),
        $image = $this.find(".image"),
        $img = $image.find("img"),
        $titleLink = $this.find(".link").not(".primary").first(),
        $primaryLink = $this.find(".link.primary").first(),
        $link,
        x;

      // Set position.
      if ((x = $img.data("position"))) $image.css("background-position", x);

      // Link.
      if ($titleLink.length > 0) {
        if ($primaryLink.length === 0) {
          $primaryLink = $titleLink
            .clone()
            .text("")
            .addClass("primary")
            .attr("aria-label", $titleLink.text().trim())
            .appendTo($this);
        }

        $link = $titleLink.add($primaryLink);

        $link.on("click", function (event) {
          var href = $titleLink.attr("href");
          var target = $titleLink.attr("target");

          // Prevent default.
          event.stopPropagation();
          event.preventDefault();

          // Start transitioning.
          $this.addClass("is-transitioning");
          $wrapper.addClass("is-transitioning");

          // Redirect.
          window.setTimeout(function () {
            if (target == "_blank") window.open(href);
            else location.href = href;
          }, 500);
        });
      }
    });

    // Header.
    if (skel.vars.IEVersion < 9) $header.removeClass("alt");

    if ($banner.length > 0 && $header.hasClass("alt")) {
      $window.on("resize", function () {
        $window.trigger("scroll");
      });

      $window.on("load", function () {
        $banner.scrollex({
          bottom: $header.height() + 10,
          terminate: function () {
            $header.removeClass("alt");
          },
          enter: function () {
            $header.addClass("alt");
          },
          leave: function () {
            $header.removeClass("alt");
            $header.addClass("reveal");
          },
        });

        window.setTimeout(function () {
          $window.triggerHandler("scroll");
        }, 100);
      });
    }

    // Menu.
    var $menu = $("#menu"),
      $menuInner;

    $menu.wrapInner('<div class="inner"></div>');
    $menuInner = $menu.children(".inner");
    $menu._locked = false;

    $menu._lock = function () {
      if ($menu._locked) return false;

      $menu._locked = true;

      window.setTimeout(function () {
        $menu._locked = false;
      }, 350);

      return true;
    };

    $menu._show = function () {
      if ($menu._lock()) $body.addClass("is-menu-visible");
    };

    $menu._hide = function () {
      if ($menu._lock()) $body.removeClass("is-menu-visible");
    };

    $menu._toggle = function () {
      if ($menu._lock()) $body.toggleClass("is-menu-visible");
    };

    $menuInner
      .on("click", function (event) {
        event.stopPropagation();
      })
      .on("click", "a", function (event) {
        var href = $(this).attr("href");

        event.preventDefault();
        event.stopPropagation();

        // Hide.
        $menu._hide();

        // Redirect.
        window.setTimeout(function () {
          window.location.href = href;
        }, 250);
      });

    $menu
      .appendTo($body)
      .on("click", function (event) {
        event.stopPropagation();
        event.preventDefault();

        $body.removeClass("is-menu-visible");
      })
      .append('<a class="close" href="#menu" aria-label="Close menu"></a>');

    $navDropdownTriggers.on("click", function (event) {
      var $dropdown = $(this).closest(".nav-dropdown");
      var isOpen = $dropdown.hasClass("is-open");

      event.preventDefault();
      event.stopPropagation();

      $navDropdowns.removeClass("is-open");
      $navDropdownTriggers.attr("aria-expanded", "false");

      if (!isOpen) {
        $dropdown.addClass("is-open");
        $(this).attr("aria-expanded", "true");
      }
    });

    $body
      .on("click", '[data-search-toggle="true"]', function (event) {
        event.preventDefault();
        event.stopPropagation();

        if (!$header.hasClass("is-search-visible")) {
          $header.addClass("is-search-visible");
          $searchToggle.attr("aria-expanded", "true");

          window.setTimeout(function () {
            $searchInput.trigger("focus");
          }, 125);
        } else {
          $header.removeClass("is-search-visible");
          $searchToggle.attr("aria-expanded", "false");
        }
      })
      .on("click", 'a[href="#menu"]', function (event) {
        event.stopPropagation();
        event.preventDefault();

        // Toggle.
        $menu._toggle();
      })
      .on("click", function (event) {
        if (
          $header.hasClass("is-search-visible") &&
          $(event.target).closest(".site-search-shell").length === 0
        ) {
          $header.removeClass("is-search-visible");
          $searchToggle.attr("aria-expanded", "false");
        }

        if ($(event.target).closest("#header .nav-dropdown").length === 0) {
          $navDropdowns.removeClass("is-open");
          $navDropdownTriggers.attr("aria-expanded", "false");
        }

        // Hide.
        $menu._hide();
      })
      .on("keydown", function (event) {
        // Hide on escape.
        if (event.keyCode == 27) {
          $menu._hide();
          $header.removeClass("is-search-visible");
          $searchToggle.attr("aria-expanded", "false");
          $navDropdowns.removeClass("is-open");
          $navDropdownTriggers.attr("aria-expanded", "false");
        }
      });

    $searchForm.on("click", function (event) {
      event.stopPropagation();
    });

    if (window.location.pathname.indexOf("/search") !== -1) {
      var searchParams = new URLSearchParams(window.location.search);
      var initialSearchQuery = searchParams.get("q");

      if (initialSearchQuery) {
        $searchInput.val(initialSearchQuery);
      }
    }
  });
})(jQuery);
