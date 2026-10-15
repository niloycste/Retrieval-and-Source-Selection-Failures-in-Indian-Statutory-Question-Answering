"""Create the retrieval, source-selection, and extraction pipeline figure."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "papers/retrieval_reference/plots/pipeline.pdf"

INK = "#152333"
BLUE = "#5b8fd1"
GREEN = "#2d9a57"
RED = "#dc5260"
AMBER = "#db9b36"


def main():
    fig, ax = plt.subplots(figsize=(7.6, 5.9))
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")
    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.02)

    def text(x, y, value, size=9.2, bold=False, ha="center", color=INK):
        ax.text(
            x, y, value, ha=ha, va="center", fontsize=size, color=color,
            weight="bold" if bold else "normal", linespacing=1.15,
        )

    def box(x, y, w, h, title, body, face="#eff8f0", edge=GREEN,
            title_size=10.4, body_size=8.8):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=.45,rounding_size=.9",
            facecolor=face, edgecolor=edge, linewidth=1.05,
        ))
        text(x + w / 2, y + h * 0.70, title, title_size, True)
        text(x + w / 2, y + h * 0.32, body, body_size)

    def arrow(x1, y1, x2, y2, dashed=False):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops={
            "arrowstyle": "-|>", "color": BLUE if dashed else "#111111",
            "lw": 0.95, "linestyle": "--" if dashed else "-",
        })

    text(27, 97, "Inference pipeline", 11.5, True)
    text(76, 97, "Offline annotation-relative checks", 10.8, True)
    ax.plot([53.5, 53.5], [5, 94], color="#a8b5c5",
            linestyle=(0, (3, 3)), linewidth=0.9)

    box(5, 82, 44, 12, "Question",
        "What does Section 30A provide\nregarding application of section 19?",
        "#eef6ff", BLUE)
    arrow(27, 82, 27, 78)

    box(5, 63, 44, 15, "1. Candidate retrieval",
        "Retrieve the top-k paragraphs with BM25,\ndense retrieval, fusion, or a section rule.")
    arrow(27, 63, 27, 59)

    box(5, 44, 44, 15, "2. Source selection",
        "Choose one candidate using reader logits\nor the independent BGE reranker.")
    arrow(27, 44, 27, 40)

    box(5, 25, 44, 15, "3. Answer extraction",
        "Use the fixed reader to extract one\ncontinuous span from the selected paragraph.")
    arrow(27, 25, 27, 21)

    box(8, 7, 38, 14, "System output",
        "Predicted answer and selected\nAct and provision.", "#eef6ff", BLUE)

    box(58, 68, 37, 21, "Candidate-access check",
        "Is the annotated paragraph in top-k?\n\nNo: candidate-access failure",
        "#fff4f5", RED)
    arrow(49, 70.5, 58, 78, True)

    box(58, 41, 37, 21, "Source-selection check",
        "Was the annotated paragraph selected?\n\nNo: source-selection error",
        "#fff4f5", RED)
    arrow(49, 51.5, 58, 51.5, True)

    box(58, 13, 37, 22, "Answer check",
        "If the annotated source was selected,\ndoes the normalised answer match?\n\nNo: non-exact extraction\nYes: source-correct exact match",
        "#fff8ea", AMBER, body_size=8.2)
    arrow(49, 32.5, 58, 24, True)

    text(76.5, 6.5,
         "Checks use the retained benchmark annotation. They do not\n"
         "alter inference or establish the legally controlling source.",
         8.0, color="#4a5663")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
