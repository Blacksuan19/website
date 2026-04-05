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
