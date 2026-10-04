"""RUS-CHN score tables and bone-age calculation."""

from typing import Literal

Sex = Literal["boy", "girl"]

BONE_ORDER = (
    "MCPFirst",
    "MCPThird",
    "MCPFifth",
    "PIPFirst",
    "PIPThird",
    "PIPFifth",
    "MIPThird",
    "MIPFifth",
    "DIPFirst",
    "DIPThird",
    "DIPFifth",
    "Ulna",
    "Radius",
)

BONE_LABELS_ZH = {
    "MCPFirst": "第一掌骨骺",
    "MCPThird": "第三掌骨骨骺",
    "MCPFifth": "第五掌骨骨骺",
    "PIPFirst": "第一近节指骨骨骺",
    "PIPThird": "第三近节指骨骨骺",
    "PIPFifth": "第五近节指骨骨骺",
    "MIPThird": "第三中节指骨骨骺",
    "MIPFifth": "第五中节指骨骨骺",
    "DIPFirst": "第一远节指骨骨骺",
    "DIPThird": "第三远节指骨骨骺",
    "DIPFifth": "第五远节指骨骨骺",
    "Ulna": "尺骨",
    "Radius": "桡骨骨骺",
}

SCORE_TABLE = {
    "girl": {
        "Radius": [10, 15, 22, 25, 40, 59, 91, 125, 138, 178, 192, 199, 203, 210],
        "Ulna": [27, 31, 36, 50, 73, 95, 120, 157, 168, 176, 182, 189],
        "MCPFirst": [5, 7, 10, 16, 23, 28, 34, 41, 47, 53, 66],
        "MCPThird": [3, 5, 6, 9, 14, 21, 32, 40, 47, 51],
        "MCPFifth": [4, 5, 7, 10, 15, 22, 33, 43, 47, 51],
        "PIPFirst": [6, 7, 8, 11, 17, 26, 32, 38, 45, 53, 60, 67],
        "PIPThird": [3, 5, 7, 9, 15, 20, 25, 29, 35, 41, 46, 51],
        "PIPFifth": [4, 5, 7, 11, 18, 21, 25, 29, 34, 40, 45, 50],
        "MIPThird": [4, 5, 7, 10, 16, 21, 25, 29, 35, 43, 46, 51],
        "MIPFifth": [3, 5, 7, 12, 19, 23, 27, 32, 35, 39, 43, 49],
        "DIPFirst": [5, 6, 8, 10, 20, 31, 38, 44, 45, 52, 67],
        "DIPThird": [3, 5, 7, 10, 16, 24, 30, 33, 36, 39, 49],
        "DIPFifth": [5, 6, 7, 11, 18, 25, 29, 33, 35, 39, 49],
    },
    "boy": {
        "Radius": [8, 11, 15, 18, 31, 46, 76, 118, 135, 171, 188, 197, 201, 209],
        "Ulna": [25, 30, 35, 43, 61, 80, 116, 157, 168, 180, 187, 194],
        "MCPFirst": [4, 5, 8, 16, 22, 26, 34, 39, 45, 52, 66],
        "MCPThird": [3, 4, 5, 8, 13, 19, 30, 38, 44, 51],
        "MCPFifth": [3, 4, 6, 9, 14, 19, 31, 41, 46, 50],
        "PIPFirst": [4, 5, 7, 11, 17, 23, 29, 36, 44, 52, 59, 66],
        "PIPThird": [3, 4, 5, 8, 14, 19, 23, 28, 34, 40, 45, 50],
        "PIPFifth": [3, 4, 6, 10, 16, 19, 24, 28, 33, 40, 44, 50],
        "MIPThird": [3, 4, 5, 9, 14, 18, 23, 28, 35, 42, 45, 50],
        "MIPFifth": [3, 4, 6, 11, 17, 21, 26, 31, 36, 40, 43, 49],
        "DIPFirst": [4, 5, 6, 9, 19, 28, 36, 43, 46, 51, 67],
        "DIPThird": [3, 4, 5, 9, 15, 23, 29, 33, 37, 40, 49],
        "DIPFifth": [3, 4, 6, 11, 17, 23, 29, 32, 36, 40, 49],
    },
}

AGE_COEFFICIENTS = {
    "boy": (
        2.01790023656577,
        -0.0931820870747269,
        0.00334709095418796,
        -3.32988302362153e-05,
        1.75712910819776e-07,
        -5.59998691223273e-10,
        1.1296711294933e-12,
        -1.45218037113138e-15,
        1.15333377080353e-18,
        -5.15887481551927e-22,
        9.94098428102335e-26,
    ),
    "girl": (
        5.81191794824917,
        -0.271546561737745,
        0.00526301486340724,
        -4.37797717401925e-05,
        2.0858722025667e-07,
        -6.21879866563429e-10,
        1.19909931745368e-12,
        -1.49462900826936e-15,
        1.162435538672e-18,
        -5.12713017846218e-22,
        9.78989966891478e-26,
    ),
}


def score_for_prediction(sex: Sex, bone_name: str, prediction_index: int) -> int:
    """Convert a zero-based classifier prediction to the corresponding RUS-CHN score."""
    try:
        scores = SCORE_TABLE[sex][bone_name]
    except KeyError as exc:
        raise ValueError(f"unsupported sex/bone combination: {sex}/{bone_name}") from exc
    if not 0 <= prediction_index < len(scores):
        raise ValueError(
            f"prediction index {prediction_index} is out of range for {bone_name} "
            f"(expected 0..{len(scores) - 1})"
        )
    return scores[prediction_index]


def calculate_bone_age(total_score: int, sex: Sex) -> float:
    """Calculate bone age from the total RUS-CHN score using Horner's method."""
    if sex not in AGE_COEFFICIENTS:
        raise ValueError("sex must be 'boy' or 'girl'")
    if total_score < 0:
        raise ValueError("total score cannot be negative")

    value = 0.0
    for coefficient in reversed(AGE_COEFFICIENTS[sex]):
        value = value * total_score + coefficient
    return round(value, 2)


def format_report(
    assessments: dict[str, tuple[int, int]], total_score: int, bone_age: float
) -> str:
    """Build the Chinese assessment report shown by the desktop UI."""
    missing = [bone for bone in BONE_ORDER if bone not in assessments]
    if missing:
        raise ValueError(f"cannot build a report with missing assessments: {', '.join(missing)}")

    lines = []
    for bone in BONE_ORDER:
        stage, score = assessments[bone]
        lines.append(f"{BONE_LABELS_ZH[bone]}分级 {stage} 级，得 {score} 分")

    detail = "；\n".join(lines) + "。"
    return (
        f"{detail}\n\n"
        f"RUS-CHN 分级计分法：CHN 总得分 {total_score} 分，估算骨龄约 {bone_age} 岁。"
    )
