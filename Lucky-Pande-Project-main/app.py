import customtkinter as ctk
import pickle
import os
import sys

# ──────────────────────────────────────────────
# Load the trained model
# ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    model = model_data["model"]
    scaler = model_data["scaler"]
    le = model_data["label_encoder"]
    feature_names = model_data["feature_names"]
    accuracy = model_data["accuracy"]
    best_k = model_data["best_k"]
except FileNotFoundError:
    print("ERROR: model.pkl not found. Run train_model.py first.")
    sys.exit(1)

# ──────────────────────────────────────────────
# Color palette & theme
# ──────────────────────────────────────────────
COLORS = {
    "bg_dark": "#0f0f1a",
    "bg_card": "#1a1a2e",
    "bg_input": "#16213e",
    "accent_purple": "#7c3aed",
    "accent_blue": "#3b82f6",
    "accent_teal": "#06b6d4",
    "text_primary": "#f1f5f9",
    "text_secondary": "#94a3b8",
    "text_muted": "#64748b",
    "success": "#10b981",
    "error": "#ef4444",
    "border": "#2d2d44",
    "setosa": "#f59e0b",
    "versicolor": "#8b5cf6",
    "virginica": "#06b6d4",
}

SPECIES_INFO = {
    "Iris-setosa": {
        "color": COLORS["setosa"],
        "emoji": "Setosa",
        "desc": "Small petals, wide sepals. Easily distinguishable from other species.",
    },
    "Iris-versicolor": {
        "color": COLORS["versicolor"],
        "emoji": "Versicolor",
        "desc": "Medium-sized petals and sepals. Intermediate characteristics.",
    },
    "Iris-virginica": {
        "color": COLORS["virginica"],
        "emoji": "Virginica",
        "desc": "Large petals with long sepals. The biggest of the three species.",
    },
}

# Feature ranges for validation (from dataset)
FEATURE_RANGES = {
    "SepalLengthCm": (4.3, 7.9),
    "SepalWidthCm": (2.0, 4.4),
    "PetalLengthCm": (1.0, 6.9),
    "PetalWidthCm": (0.1, 2.5),
}

FEATURE_LABELS = {
    "SepalLengthCm": "Sepal Length (cm)",
    "SepalWidthCm": "Sepal Width (cm)",
    "PetalLengthCm": "Petal Length (cm)",
    "PetalWidthCm": "Petal Width (cm)",
}


# ──────────────────────────────────────────────
# Application class
# ──────────────────────────────────────────────
class IrisClassifierApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window config
        self.title("Iris Flower Classifier")
        self.geometry("520x720")
        self.minsize(420, 500)
        self.resizable(True, True)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.configure(fg_color=COLORS["bg_dark"])

        self.entries = {}
        self._build_ui()

    def _build_ui(self):
        # ── Main scrollable container ──
        container = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            scrollbar_button_color=COLORS["border"],
            scrollbar_button_hover_color=COLORS["accent_purple"],
        )
        container.pack(fill="both", expand=True, padx=20, pady=20)

        # ── Header ──
        header_frame = ctk.CTkFrame(container, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 8))

        title_label = ctk.CTkLabel(
            header_frame,
            text="Iris Flower Classifier",
            font=ctk.CTkFont(family="Segoe UI", size=26, weight="bold"),
            text_color=COLORS["text_primary"],
        )
        title_label.pack(anchor="w")

        subtitle_label = ctk.CTkLabel(
            header_frame,
            text=f"KNN Model (K={best_k})  |  Accuracy: {accuracy * 100:.1f}%",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=COLORS["text_muted"],
        )
        subtitle_label.pack(anchor="w", pady=(2, 0))

        # Accent line
        accent_line = ctk.CTkFrame(
            container, height=3, fg_color=COLORS["accent_purple"], corner_radius=2
        )
        accent_line.pack(fill="x", pady=(0, 16))

        # ── Input Card ──
        input_card = ctk.CTkFrame(
            container,
            fg_color=COLORS["bg_card"],
            corner_radius=14,
            border_width=1,
            border_color=COLORS["border"],
        )
        input_card.pack(fill="x", pady=(0, 12))

        input_header = ctk.CTkLabel(
            input_card,
            text="Enter Measurements",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            text_color=COLORS["text_primary"],
        )
        input_header.pack(anchor="w", padx=20, pady=(16, 12))

        # Feature inputs (2x2 grid)
        grid_frame = ctk.CTkFrame(input_card, fg_color="transparent")
        grid_frame.pack(fill="x", padx=20, pady=(0, 16))
        grid_frame.columnconfigure(0, weight=1)
        grid_frame.columnconfigure(1, weight=1)

        for i, feat in enumerate(feature_names):
            row, col = divmod(i, 2)
            self._create_input_field(grid_frame, feat, row, col)

        # ── Predict Button ──
        predict_btn = ctk.CTkButton(
            container,
            text="Predict Species",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            height=46,
            corner_radius=12,
            fg_color=COLORS["accent_purple"],
            hover_color="#6d28d9",
            command=self._predict,
        )
        predict_btn.pack(fill="x", pady=(4, 12))

        # ── Result Card ──
        self.result_card = ctk.CTkFrame(
            container,
            fg_color=COLORS["bg_card"],
            corner_radius=14,
            border_width=1,
            border_color=COLORS["border"],
        )
        self.result_card.pack(fill="x", pady=(0, 12))

        result_header = ctk.CTkLabel(
            self.result_card,
            text="Prediction Result",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            text_color=COLORS["text_primary"],
        )
        result_header.pack(anchor="w", padx=20, pady=(16, 8))

        # Placeholder text
        self.result_placeholder = ctk.CTkLabel(
            self.result_card,
            text="Enter measurements and click Predict",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=COLORS["text_muted"],
        )
        self.result_placeholder.pack(padx=20, pady=(0, 16))

        # Result content frame (hidden initially)
        self.result_content = ctk.CTkFrame(self.result_card, fg_color="transparent")

        # Species name
        self.species_label = ctk.CTkLabel(
            self.result_content,
            text="",
            font=ctk.CTkFont(family="Segoe UI", size=28, weight="bold"),
        )
        self.species_label.pack(anchor="w", padx=20, pady=(0, 4))

        # Description
        self.desc_label = ctk.CTkLabel(
            self.result_content,
            text="",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=COLORS["text_secondary"],
            wraplength=440,
            justify="left",
        )
        self.desc_label.pack(anchor="w", padx=20, pady=(0, 12))

        # Confidence bar section
        self.confidence_frame = ctk.CTkFrame(
            self.result_content, fg_color="transparent"
        )
        self.confidence_frame.pack(fill="x", padx=20, pady=(0, 24))

        # ── Error label (hidden) ──
        self.error_label = ctk.CTkLabel(
            container,
            text="",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=COLORS["error"],
        )
        self.error_label.pack(fill="x")

        # ── Clear Button ──
        clear_btn = ctk.CTkButton(
            container,
            text="Clear All",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            height=36,
            corner_radius=10,
            fg_color="transparent",
            hover_color=COLORS["bg_input"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["text_secondary"],
            command=self._clear,
        )
        clear_btn.pack(fill="x", pady=(0, 0))

    def _create_input_field(self, parent, feature, row, col):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=col, padx=(0, 10 if col == 0 else 0), pady=6, sticky="ew")

        min_val, max_val = FEATURE_RANGES[feature]

        label = ctk.CTkLabel(
            frame,
            text=FEATURE_LABELS[feature],
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=COLORS["text_secondary"],
        )
        label.pack(anchor="w")

        entry = ctk.CTkEntry(
            frame,
            height=38,
            corner_radius=8,
            fg_color=COLORS["bg_input"],
            border_color=COLORS["border"],
            border_width=1,
            text_color=COLORS["text_primary"],
            placeholder_text=f"{min_val} - {max_val}",
            placeholder_text_color=COLORS["text_muted"],
            font=ctk.CTkFont(family="Segoe UI", size=13),
        )
        entry.pack(fill="x", pady=(4, 0))

        self.entries[feature] = entry

    def _predict(self):
        self.error_label.configure(text="")

        # Validate inputs
        values = []
        for feat in feature_names:
            raw = self.entries[feat].get().strip()
            if not raw:
                self.error_label.configure(
                    text=f"Please enter a value for {FEATURE_LABELS[feat]}"
                )
                return
            try:
                val = float(raw)
            except ValueError:
                self.error_label.configure(
                    text=f"Invalid number for {FEATURE_LABELS[feat]}"
                )
                return

            min_v, max_v = FEATURE_RANGES[feat]
            if val < min_v - 2 or val > max_v + 2:
                self.error_label.configure(
                    text=f"{FEATURE_LABELS[feat]}: value {val} is far outside training range ({min_v}-{max_v})"
                )
                return
            values.append(val)

        # Scale and predict
        import numpy as np

        input_array = np.array([values])
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        species_name = le.inverse_transform([prediction])[0]

        # Get probabilities (KNN uses neighbor voting)
        distances, indices = model.kneighbors(input_scaled)
        neighbor_labels = model._y[indices[0]]
        probs = {}
        for cls_idx, cls_name in enumerate(le.classes_):
            count = (neighbor_labels == cls_idx).sum()
            probs[cls_name] = count / len(neighbor_labels)

        # Update UI
        self._show_result(species_name, probs)

    def _show_result(self, species, probabilities):
        info = SPECIES_INFO[species]

        # Hide placeholder, show content
        self.result_placeholder.pack_forget()
        self.result_content.pack(fill="x")

        # Update border color of result card
        self.result_card.configure(border_color=info["color"])

        # Species name
        self.species_label.configure(text=info["emoji"], text_color=info["color"])

        # Description
        self.desc_label.configure(text=info["desc"])

        # Rebuild confidence bars
        for widget in self.confidence_frame.winfo_children():
            widget.destroy()

        conf_title = ctk.CTkLabel(
            self.confidence_frame,
            text="Neighbor Voting",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=COLORS["text_secondary"],
        )
        conf_title.pack(anchor="w", pady=(0, 8))

        for cls_name in le.classes_:
            prob = probabilities.get(cls_name, 0)
            cls_info = SPECIES_INFO[cls_name]

            row_frame = ctk.CTkFrame(self.confidence_frame, fg_color="transparent")
            row_frame.pack(fill="x", pady=3)

            name_label = ctk.CTkLabel(
                row_frame,
                text=cls_name.replace("Iris-", ""),
                font=ctk.CTkFont(family="Segoe UI", size=12),
                text_color=COLORS["text_secondary"],
                width=80,
                anchor="w",
            )
            name_label.pack(side="left")

            bar_bg = ctk.CTkFrame(
                row_frame,
                height=14,
                fg_color=COLORS["bg_input"],
                corner_radius=7,
            )
            bar_bg.pack(side="left", fill="x", expand=True, padx=(8, 8))
            bar_bg.update_idletasks()

            if prob > 0:
                bar_fill = ctk.CTkFrame(
                    bar_bg,
                    height=14,
                    fg_color=cls_info["color"],
                    corner_radius=7,
                )
                bar_fill.place(relx=0, rely=0, relwidth=max(prob, 0.03), relheight=1.0)

            pct_label = ctk.CTkLabel(
                row_frame,
                text=f"{prob * 100:.0f}%",
                font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
                text_color=cls_info["color"] if prob > 0 else COLORS["text_muted"],
                width=45,
                anchor="e",
            )
            pct_label.pack(side="right")

    def _clear(self):
        for entry in self.entries.values():
            entry.delete(0, "end")

        self.error_label.configure(text="")

        # Reset result card
        self.result_content.pack_forget()
        self.result_placeholder.pack(padx=20, pady=(0, 16))
        self.result_card.configure(border_color=COLORS["border"])


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app = IrisClassifierApp()
    app.mainloop()
