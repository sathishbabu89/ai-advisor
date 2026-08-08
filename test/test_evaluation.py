import logging

from app.evaluation.evaluator import Evaluator


logging.basicConfig(
    level=logging.INFO
)


def main():

    evaluator = Evaluator()


    question = (
        "What are the benefits of SIP?"
    )


    context = [

        """
        Major Benefits of SIP

        - Rupee Cost Averaging
        - Power of Compounding
        - Automated Discipline
        - High Flexibility
        - Potential Tax Advantages
        """
    ]


    answer = """
    The benefits of SIP include:

    1. Rupee Cost Averaging
    2. Power of Compounding
    3. Automated Discipline
    4. High Flexibility
    5. Potential Tax Advantages
    """

    result = evaluator.evaluate_response(
        question,
        answer,
        context
    )


    print(result)


if __name__ == "__main__":
    main()