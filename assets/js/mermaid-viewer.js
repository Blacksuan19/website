(function () {
  var DIAGRAM_SELECTOR =
    "pre code.language-mermaid, .language-mermaid code, code.language-mermaid";
  var PREVIEW_MAX_SCALE = 2.2;
  var SCALE_STEP = 0.2;

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function roundScale(value) {
    return Math.round(value * 100) / 100;
  }

  function nearlyEqual(left, right) {
    return Math.abs(left - right) < 0.01;
  }

  function getCssVariable(name, fallback) {
    var value = getComputedStyle(document.documentElement)
      .getPropertyValue(name)
      .trim();

    return value || fallback;
  }

  function findReplacementTarget(element) {
    var pre = element.closest("pre");

    if (!pre) {
      return null;
    }

    return pre.closest(".highlighter-rouge") || pre.closest(".highlight") || pre;
  }

  function getSvgDimensions(svg) {
    var viewBox = svg.viewBox && svg.viewBox.baseVal;

    if (viewBox && viewBox.width && viewBox.height) {
      return {
        width: viewBox.width,
        height: viewBox.height,
      };
    }

    var rect = svg.getBBox ? svg.getBBox() : svg.getBoundingClientRect();

    return {
      width: rect.width || 800,
      height: rect.height || 600,
    };
  }

  function getViewportPadding(viewport) {
    var styles = getComputedStyle(viewport);

    return {
      horizontal:
        parseFloat(styles.paddingLeft || "0") +
        parseFloat(styles.paddingRight || "0"),
      vertical:
        parseFloat(styles.paddingTop || "0") +
        parseFloat(styles.paddingBottom || "0"),
    };
  }

  function findNearestPreviousHeading(element) {
    var current = element;

    while (current) {
      var sibling = current.previousElementSibling;

      while (sibling) {
        if (/^H[1-6]$/.test(sibling.tagName)) {
          return sibling;
        }

        var nestedHeading = sibling.querySelector
          ? sibling.querySelector("h1, h2, h3, h4, h5, h6")
          : null;

        if (nestedHeading) {
          return nestedHeading;
        }

        sibling = sibling.previousElementSibling;
      }

      current = current.parentElement;
    }

    return null;
  }

  function inferDiagramTitle(target, index) {
    var heading = findNearestPreviousHeading(target);

    if (heading) {
      return heading.textContent.trim();
    }

    return "Diagram " + (index + 1);
  }

  function syncButtonState(viewer) {
    viewer.controls.zoomOut.disabled = viewer.scale <= viewer.minScale;
    viewer.controls.zoomIn.disabled = viewer.scale >= viewer.maxScale;
    viewer.controls.reset.disabled = nearlyEqual(viewer.scale, viewer.defaultScale);
  }

  function measureControlWidth(button) {
    var clone = button.cloneNode(true);
    var width;

    clone.disabled = false;
    clone.style.left = "-9999px";
    clone.style.minWidth = "0";
    clone.style.position = "absolute";
    clone.style.top = "0";
    clone.style.visibility = "hidden";
    clone.style.width = "auto";

    document.body.appendChild(clone);
    width = Math.ceil(clone.getBoundingClientRect().width);
    document.body.removeChild(clone);

    return width;
  }

  function syncControlWidth(viewer) {
    var buttons;
    var maxWidth;

    if (window.matchMedia("(max-width: 736px)").matches) {
      viewer.root.style.removeProperty("--mermaid-control-width");
      return;
    }

    buttons = [
      viewer.controls.zoomOut,
      viewer.controls.reset,
      viewer.controls.zoomIn,
    ];
    maxWidth = buttons.reduce(function (currentMax, button) {
      return Math.max(currentMax, measureControlWidth(button));
    }, 0);

    viewer.root.style.setProperty("--mermaid-control-width", maxWidth + "px");
  }

  function updateScale(viewer, nextScale) {
    var viewport = viewer.viewport;
    var previousScale = viewer.scale || viewer.defaultScale || 1;
    var centerX = viewport.scrollLeft + viewport.clientWidth / 2;
    var centerY = viewport.scrollTop + viewport.clientHeight / 2;
    var ratio = nextScale / previousScale;
    var viewportPadding = getViewportPadding(viewport);
    var availableWidth = Math.max(viewport.clientWidth - viewportPadding.horizontal, 220);
    var availableHeight = Math.max(viewport.clientHeight - viewportPadding.vertical, 180);
    var contentWidth;
    var contentHeight;

    viewer.scale = roundScale(nextScale);
    contentWidth = Math.round(viewer.baseWidth * viewer.scale);
    contentHeight = Math.round(viewer.baseHeight * viewer.scale);

    viewer.canvas.style.width = Math.max(contentWidth, availableWidth) + "px";
    viewer.canvas.style.height = Math.max(contentHeight, availableHeight) + "px";
    viewer.svg.style.width = contentWidth + "px";
    viewer.svg.style.height = contentHeight + "px";
    viewer.scaleLabel.textContent = Math.round(viewer.scale * 100) + "%";
    syncButtonState(viewer);
    syncControlWidth(viewer);

    viewport.scrollLeft = Math.max(0, centerX * ratio - viewport.clientWidth / 2);
    viewport.scrollTop = Math.max(0, centerY * ratio - viewport.clientHeight / 2);
  }

  function computeFitMetrics(viewport, baseWidth, baseHeight) {
    if (!viewport || !baseWidth || !baseHeight) {
      return {
        fitScale: 1,
      };
    }

    var viewportPadding = getViewportPadding(viewport);
    var availableWidth = Math.max(viewport.clientWidth - viewportPadding.horizontal, 220);
    var availableHeight = Math.max(viewport.clientHeight - viewportPadding.vertical, 180);
    var widthScale = availableWidth / baseWidth;
    var heightScale = availableHeight / baseHeight;
    var fitScale = roundScale(Math.min(widthScale, heightScale, 1));

    return {
      fitScale: fitScale,
    };
  }

  function computeReadableDefaultScale(viewport, baseWidth, baseHeight, fitScale) {
    var viewportPadding;
    var availableHeight;
    var aspectRatio;
    var targetHeight;
    var readableScale;

    if (!viewport || !baseWidth || !baseHeight) {
      return fitScale || 1;
    }

    viewportPadding = getViewportPadding(viewport);
    availableHeight = Math.max(viewport.clientHeight - viewportPadding.vertical, 180);
    aspectRatio = baseWidth / Math.max(baseHeight, 1);

    if (aspectRatio < 2) {
      return fitScale;
    }

    targetHeight = Math.max(Math.min(availableHeight * 0.42, 260), 150);
    readableScale = roundScale(Math.min(targetHeight / baseHeight, 1));

    return Math.max(fitScale, readableScale);
  }

  function createViewerState(root, viewport, render, svg, scaleLabel, controls) {
    var dimensions = getSvgDimensions(svg);
    var fit = computeFitMetrics(viewport, dimensions.width, dimensions.height);
    var defaultScale = computeReadableDefaultScale(
      viewport,
      dimensions.width,
      dimensions.height,
      fit.fitScale
    );

    return {
      root: root,
      viewport: viewport,
      canvas: viewport.querySelector(".mermaid-diagram__canvas"),
      svg: svg,
      scaleLabel: scaleLabel,
      controls: controls,
      minScale: fit.fitScale,
      maxScale: PREVIEW_MAX_SCALE,
      baseWidth: dimensions.width,
      baseHeight: dimensions.height,
      defaultScale: defaultScale,
      scale: defaultScale,
    };
  }

  function setViewerScale(viewer, nextScale) {
    updateScale(viewer, clamp(nextScale, viewer.minScale, viewer.maxScale));
  }

  function buildPreviewShell(source, index, title) {
    var figure = document.createElement("figure");
    var toolbar = document.createElement("div");
    var actions = document.createElement("div");
    var zoomOutButton = document.createElement("button");
    var scaleButton = document.createElement("button");
    var zoomInButton = document.createElement("button");
    var titleLabel = document.createElement("p");
    var viewport = document.createElement("div");
    var canvas = document.createElement("div");
    var render = document.createElement("div");

    figure.className = "mermaid-diagram";
    figure.dataset.mermaidIndex = String(index);

    toolbar.className = "mermaid-diagram__toolbar";
    actions.className = "mermaid-diagram__actions";

    zoomOutButton.className = "mermaid-diagram__button";
    zoomOutButton.dataset.mermaidAction = "zoom-out";
    zoomOutButton.type = "button";
    zoomOutButton.setAttribute("aria-label", "Zoom out");
    zoomOutButton.textContent = "-";

    scaleButton.className = "mermaid-diagram__button";
    scaleButton.dataset.mermaidAction = "reset";
    scaleButton.dataset.mermaidScale = "true";
    scaleButton.type = "button";
    scaleButton.setAttribute("aria-label", "Reset zoom");
    scaleButton.textContent = "100%";

    zoomInButton.className = "mermaid-diagram__button";
    zoomInButton.dataset.mermaidAction = "zoom-in";
    zoomInButton.type = "button";
    zoomInButton.setAttribute("aria-label", "Zoom in");
    zoomInButton.textContent = "+";

    titleLabel.className = "mermaid-diagram__title";
    titleLabel.textContent = title;

    viewport.className = "mermaid-diagram__viewport";
    canvas.className = "mermaid-diagram__canvas";
    render.className = "mermaid mermaid-diagram__render";
    render.id = "mermaid-diagram-" + index;
    render.textContent = source;

    actions.appendChild(zoomOutButton);
    actions.appendChild(scaleButton);
    actions.appendChild(zoomInButton);
    toolbar.appendChild(actions);
    toolbar.appendChild(titleLabel);
    canvas.appendChild(render);
    viewport.appendChild(canvas);
    figure.appendChild(toolbar);
    figure.appendChild(viewport);

    return figure;
  }

  function replaceCodeBlocks() {
    var elements = document.querySelectorAll(DIAGRAM_SELECTOR);
    var targets = new Set();
    var diagrams = [];

    elements.forEach(function (element) {
      var target = findReplacementTarget(element);
      var source = element.textContent.trim();

      if (!target || !source || targets.has(target)) {
        return;
      }

      targets.add(target);
      diagrams.push({
        target: target,
        source: source,
        title: inferDiagramTitle(target, diagrams.length),
      });
    });

    diagrams.forEach(function (diagram, index) {
      diagram.target.replaceWith(buildPreviewShell(diagram.source, index, diagram.title));
    });
  }

  function createPreviewViewers() {
    return Array.prototype.map.call(
      document.querySelectorAll(".mermaid-diagram"),
      function (root) {
        var viewport = root.querySelector(".mermaid-diagram__viewport");
        var render = root.querySelector(".mermaid-diagram__render");
        var svg = render.querySelector("svg");
        var viewer = createViewerState(
          root,
          viewport,
          render,
          svg,
          root.querySelector("[data-mermaid-scale]"),
          {
            zoomOut: root.querySelector('[data-mermaid-action="zoom-out"]'),
            reset: root.querySelector('[data-mermaid-action="reset"]'),
            zoomIn: root.querySelector('[data-mermaid-action="zoom-in"]'),
          }
        );

        render.dataset.mermaidReady = "true";
        setViewerScale(viewer, viewer.defaultScale);
        root.__mermaidViewer = viewer;

        return viewer;
      }
    );
  }

  function bindPreviewActions() {
    document.addEventListener("click", function (event) {
      var button = event.target.closest("[data-mermaid-action]");
      var root;
      var viewer;

      if (!button) {
        return;
      }

      root = button.closest(".mermaid-diagram");
      viewer = root && root.__mermaidViewer;

      if (!viewer) {
        return;
      }

      switch (button.dataset.mermaidAction) {
        case "zoom-in":
          setViewerScale(viewer, viewer.scale + SCALE_STEP);
          break;
        case "zoom-out":
          setViewerScale(viewer, viewer.scale - SCALE_STEP);
          break;
        case "reset":
          setViewerScale(viewer, viewer.defaultScale);
          break;
      }
    });

    window.addEventListener("resize", function () {
      document.querySelectorAll(".mermaid-diagram").forEach(function (root) {
        var viewer = root.__mermaidViewer;
        var fit;

        if (!viewer) {
          return;
        }

        fit = computeFitMetrics(viewer.viewport, viewer.baseWidth, viewer.baseHeight);
        viewer.minScale = fit.fitScale;
        viewer.defaultScale = computeReadableDefaultScale(
          viewer.viewport,
          viewer.baseWidth,
          viewer.baseHeight,
          fit.fitScale
        );
        viewer.maxScale = Math.max(viewer.minScale + SCALE_STEP, PREVIEW_MAX_SCALE);

        if (viewer.scale <= viewer.defaultScale + SCALE_STEP / 2) {
          setViewerScale(viewer, viewer.defaultScale);
        } else {
          setViewerScale(viewer, viewer.scale);
        }
      });
    });
  }

  function getMermaidTheme() {
    return {
      theme: "base",
      darkMode: true,
      primaryColor: getCssVariable("--mermaid-theme-surface", "#181b26"),
      primaryTextColor: getCssVariable("--mermaid-theme-text", "#ffffff"),
      primaryBorderColor: getCssVariable("--mermaid-theme-accent", "#9bf1ff"),
      lineColor: getCssVariable("--mermaid-theme-accent", "#9bf1ff"),
      secondaryColor: getCssVariable("--mermaid-theme-accent-soft", "#6fc3df"),
      tertiaryColor: getCssVariable("--mermaid-theme-surface-alt", "rgba(255,255,255,0.04)"),
      background: getCssVariable("--mermaid-theme-bg", "#0f111a"),
      mainBkg: getCssVariable("--mermaid-theme-surface", "#181b26"),
      secondBkg: getCssVariable("--mermaid-theme-surface-alt", "rgba(255,255,255,0.04)"),
      tertiaryBkg: getCssVariable("--mermaid-theme-bg", "#0f111a"),
      clusterBkg: getCssVariable("--mermaid-theme-surface-alt", "rgba(255,255,255,0.04)"),
      clusterBorder: getCssVariable("--mermaid-theme-border", "rgba(255,255,255,0.12)"),
      edgeLabelBackground: getCssVariable("--mermaid-theme-bg", "#0f111a"),
      defaultLinkColor: getCssVariable("--mermaid-theme-accent", "#9bf1ff"),
      titleColor: getCssVariable("--mermaid-theme-text", "#ffffff"),
      textColor: getCssVariable("--mermaid-theme-text", "#ffffff"),
      actorBkg: getCssVariable("--mermaid-theme-surface", "#181b26"),
      actorBorder: getCssVariable("--mermaid-theme-accent", "#9bf1ff"),
      actorTextColor: getCssVariable("--mermaid-theme-text", "#ffffff"),
      actorLineColor: getCssVariable("--mermaid-theme-border", "rgba(255,255,255,0.12)"),
      signalColor: getCssVariable("--mermaid-theme-text", "#ffffff"),
      signalTextColor: getCssVariable("--mermaid-theme-text", "#ffffff"),
      labelBoxBkgColor: getCssVariable("--mermaid-theme-bg", "#0f111a"),
      labelBoxBorderColor: getCssVariable("--mermaid-theme-border", "rgba(255,255,255,0.12)"),
      labelTextColor: getCssVariable("--mermaid-theme-text", "#ffffff"),
      fontFamily: "Source Sans Pro, Helvetica, sans-serif",
    };
  }

  function initializeMermaid() {
    mermaid.initialize({
      startOnLoad: false,
      theme: "base",
      securityLevel: "loose",
      themeVariables: getMermaidTheme(),
      flowchart: {
        useMaxWidth: false,
        htmlLabels: true,
        nodeSpacing: 150,
        rankSpacing: 200,
        curve: "basis",
      },
      classDiagram: {
        useMaxWidth: false,
        htmlLabels: true,
        nodeSpacing: 150,
        rankSpacing: 200,
      },
      fontSize: 18,
      fontFamily: "Source Sans Pro, Helvetica, sans-serif",
      themeCSS: [
        ".node rect, .node circle, .node ellipse, .node polygon, .node path { stroke-width: 2px; rx: 12px; ry: 12px; }",
        ".edgePath .path, .relationshipLine { stroke-width: 2px; }",
        ".nodeLabel, .label text, .cluster-label text, .edgeLabel text, .messageText, .label foreignObject span { color: " +
        getCssVariable("--mermaid-theme-text", "#ffffff") +
        "; fill: " +
        getCssVariable("--mermaid-theme-text", "#ffffff") +
        "; }",
        ".edgeLabel rect, .labelBkg { fill: " +
        getCssVariable("--mermaid-theme-bg", "#0f111a") +
        "; opacity: 0.92; }",
        ".cluster rect, .classGroup rect { rx: 14px; ry: 14px; }",
        ".classBox, .actor, .labelBox { stroke-width: 2px; }",
        ".marker { fill: " + getCssVariable("--mermaid-theme-accent", "#9bf1ff") + "; }",
        ".activation0, .activation1, .activation2 { fill: " +
        getCssVariable("--mermaid-theme-surface-alt", "rgba(255,255,255,0.04)") +
        "; stroke: " +
        getCssVariable("--mermaid-theme-accent", "#9bf1ff") +
        "; }",
      ].join("\n"),
    });
  }

  function renderPreviews() {
    return mermaid.run({
      nodes: Array.prototype.slice.call(
        document.querySelectorAll(".mermaid-diagram__render")
      ),
    });
  }

  function initialize() {
    if (typeof mermaid === "undefined") {
      return;
    }

    replaceCodeBlocks();

    if (!document.querySelector(".mermaid-diagram__render")) {
      return;
    }

    initializeMermaid();
    bindPreviewActions();
    renderPreviews()
      .then(function () {
        createPreviewViewers();
      })
      .catch(function (error) {
        console.error("Failed to render Mermaid diagrams", error);
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize);
  } else {
    initialize();
  }
})();