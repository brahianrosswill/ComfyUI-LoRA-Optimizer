import { app } from "/scripts/app.js";

const HIDDEN_TAG = "loraopt_hidden";
const origProps = {};

function toggleWidget(node, widget, show, suffix = "") {
    if (!widget) return;

    if (!origProps[widget.name]) {
        origProps[widget.name] = {
            origType: widget.type,
            origComputeSize: widget.computeSize,
        };
    }

    widget.hidden = !show;
    widget.type = show ? origProps[widget.name].origType : HIDDEN_TAG + suffix;
    widget.computeSize = show
        ? origProps[widget.name].origComputeSize
        : () => [0, -4];

    if (widget.linkedWidgets) {
        for (const w of widget.linkedWidgets) {
            toggleWidget(node, w, show, ":" + widget.name);
        }
    }
}

function findWidget(node, name) {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}

function updateVisibility(node) {
    const modeWidget = findWidget(node, "mode");
    const countWidget = findWidget(node, "lora_count");
    if (!modeWidget || !countWidget) return;

    const isSimple = modeWidget.value === "simple";
    const count = countWidget.value;
    const MAX = 10;

    for (let i = 1; i <= MAX; i++) {
        const visible = i <= count;

        toggleWidget(node, findWidget(node, `lora_name_${i}`), visible);
        toggleWidget(node, findWidget(node, `strength_${i}`), visible && isSimple);
        toggleWidget(node, findWidget(node, `model_strength_${i}`), visible && !isSimple);
        toggleWidget(node, findWidget(node, `clip_strength_${i}`), visible && !isSimple);
        toggleWidget(node, findWidget(node, `conflict_mode_${i}`), visible);
    }

    const newHeight = node.computeSize()[1];
    node.setSize([node.size[0], newHeight]);
    app.canvas?.setDirty?.(true, true);
}

// --- Base Model Filter (requires ComfyUI-Lora-Manager) ---

const LM_BASE_MODELS_URL = "/api/lm/loras/base-models?limit=100";
const LM_LORAS_LIST_URL = "/api/lm/loras/list";

async function fetchJson(url) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return resp.json();
}

function setComboOptions(widget, options) {
    widget.options.values = options;
    if (options.length > 0 && !options.includes(widget.value)) {
        widget.value = options[0];
    }
}

async function initBaseModelFilter(node) {
    const filterWidget = findWidget(node, "base_model_filter");
    if (!filterWidget) return;

    // Cache the full LoRA list from the first lora_name widget
    const firstLoraWidget = findWidget(node, "lora_name_1");
    if (!firstLoraWidget) return;
    const fullLoraList = [...(firstLoraWidget.options.values || [])];

    // Try to detect Lora Manager
    let baseModels;
    try {
        const data = await fetchJson(LM_BASE_MODELS_URL);
        baseModels = (data.base_models || []).map((m) => m.name).filter(Boolean);
    } catch {
        // Lora Manager not installed — hide filter widget
        toggleWidget(node, filterWidget, false);
        updateVisibility(node);
        return;
    }

    if (baseModels.length === 0) {
        toggleWidget(node, filterWidget, false);
        updateVisibility(node);
        return;
    }

    // Populate filter dropdown
    const filterOptions = ["All", ...baseModels];
    setComboOptions(filterWidget, filterOptions);

    // Intercept filter changes
    let filterValue = filterWidget.value;
    const desc =
        Object.getOwnPropertyDescriptor(filterWidget, "value") ||
        Object.getOwnPropertyDescriptor(
            Object.getPrototypeOf(filterWidget),
            "value"
        );

    Object.defineProperty(filterWidget, "value", {
        get() {
            return desc?.get ? desc.get.call(filterWidget) : filterValue;
        },
        set(newVal) {
            if (desc?.set) {
                desc.set.call(filterWidget, newVal);
            } else {
                filterValue = newVal;
            }
            applyLoraFilter(node, newVal, fullLoraList);
        },
    });
}

async function applyLoraFilter(node, baseModel, fullLoraList) {
    const MAX = 10;
    let filteredList;

    if (baseModel === "All") {
        filteredList = fullLoraList;
    } else {
        try {
            const params = new URLSearchParams({
                base_model: baseModel,
                page_size: "10000",
                sort_by: "name",
            });
            const data = await fetchJson(`${LM_LORAS_LIST_URL}?${params}`);
            const paths = (data.items || []).map((item) => item.file_path).filter(Boolean);
            // Keep "None" at the top, then filtered paths
            filteredList = ["None", ...paths];
        } catch {
            filteredList = fullLoraList;
        }
    }

    for (let i = 1; i <= MAX; i++) {
        const w = findWidget(node, `lora_name_${i}`);
        if (w) setComboOptions(w, filteredList);
    }

    app.canvas?.setDirty?.(true, true);
}

// --- Node Registration ---

app.registerExtension({
    name: "LoRAOptimizer.LoRAStackDynamic",
    nodeCreated(node) {
        if (node.comfyClass !== "LoRAStackDynamic") return;

        // Intercept mode and lora_count changes to update visibility
        for (const w of node.widgets || []) {
            if (w.name !== "mode" && w.name !== "lora_count") continue;

            let widgetValue = w.value;
            const originalDescriptor =
                Object.getOwnPropertyDescriptor(w, "value") ||
                Object.getOwnPropertyDescriptor(
                    Object.getPrototypeOf(w),
                    "value"
                );

            Object.defineProperty(w, "value", {
                get() {
                    return originalDescriptor?.get
                        ? originalDescriptor.get.call(w)
                        : widgetValue;
                },
                set(newVal) {
                    if (originalDescriptor?.set) {
                        originalDescriptor.set.call(w, newVal);
                    } else {
                        widgetValue = newVal;
                    }
                    updateVisibility(node);
                },
            });
        }

        // Initial visibility update — delay to ensure widgets are fully initialized
        setTimeout(() => updateVisibility(node), 100);

        // Initialize base model filter (async, non-blocking)
        setTimeout(() => initBaseModelFilter(node), 200);
    },
});

app.registerExtension({
    name: "LoRAOptimizer.LoRAConflictEditor",
    nodeCreated(node) {
        if (node.comfyClass !== "LoRAConflictEditor") return;
        // All 10 conflict_mode slots are always visible.
        // Unused slots (beyond the stack size) default to "auto" and are ignored.
    },
});
