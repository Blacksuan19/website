$(document).ready(function () {
  $("[data-share]").each(function () {
    var shareRoot = this;
    var shareButton = shareRoot.querySelector("[data-share-trigger]");
    var shareTitle = shareRoot.getAttribute("data-share-title") || document.title;
    var shareUrl = shareRoot.getAttribute("data-share-url") || window.location.href;
    var canUseNativeShare = typeof navigator.share === "function";

    if (!shareButton) {
      return;
    }

    if (!canUseNativeShare) {
      shareRoot.setAttribute("hidden", "hidden");
      return;
    }

    shareButton.removeAttribute("hidden");

    shareButton.addEventListener("click", function () {
      navigator.share({
        title: shareTitle,
        url: shareUrl,
      }).catch(function (error) {
        if (error && error.name === "AbortError") {
          return;
        }

        shareRoot.setAttribute("hidden", "hidden");
      });
    });
  });

  $("[data-code-copy]").each(function () {
    var button = this;
    var resetTimer;
    var defaultLabel = button.getAttribute("aria-label") || "Copy code";

    button.addEventListener("click", function () {
      var root = button.closest("[data-code-block]");
      var code = root ? root.querySelector(".shiki code") : null;

      if (!code || !navigator.clipboard || typeof navigator.clipboard.writeText !== "function") {
        return;
      }

      navigator.clipboard.writeText(code.textContent || "").then(function () {
        window.clearTimeout(resetTimer);
        button.setAttribute("aria-label", "Copied");
        button.setAttribute("title", "Copied");
        button.classList.add("is-copied");
        resetTimer = window.setTimeout(function () {
          button.setAttribute("aria-label", defaultLabel);
          button.setAttribute("title", defaultLabel);
          button.classList.remove("is-copied");
        }, 1800);
      });
    });
  });

  // back to top button
  $(window).scroll(function () {
    if ($(this).scrollTop() > 100) {
      $("#scroll").fadeIn();
    } else {
      $("#scroll").fadeOut();
    }
  });
  $("#scroll").click(function () {
    $("html, body").animate({ scrollTop: 0 }, 600);
    return false;
  });

  // logic for collapsable Headers
  $(".col_head").click(function () {
    var current = $(this).nextAll("div .col_con").first();
    $(this)
      .parent()
      .find(".col_con")
      .each(function () {
        if (current.is(this)) {
          $(this).slideToggle("fast");
        } else {
          $(this).slideUp("fast");
        }
      });
  });

  // contact form validation
  $(document).ready(function () {
    $("#submit-button").click(function () {
      $("#contact-form").validate({
        rules: {
          name: {
            required: true,
          },
          email: {
            required: true,
            email: true,
          },
          message: {
            required: true,
          },
        },
        submitHandler: function () {
          return true;
        },
      });
    });
  });
});
