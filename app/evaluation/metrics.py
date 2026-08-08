from deepeval.metrics import FaithfulnessMetric

from app.evaluation.judge_model import GroqJudgeModel


def get_rag_metrics():

    judge_model = GroqJudgeModel()


    return [

        FaithfulnessMetric(
            threshold=0.7,
            model=judge_model
        )

    ]