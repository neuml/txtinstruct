"""
Instructor module
"""

from string import Formatter

from datasets import Dataset

from txtai.pipeline import HFTrainer


class Instructor:
    """
    Trains a model using an instruction-tuning dataset.
    """

    def __call__(self, base, data, task, prompt=None, **kwargs):
        """
        Trains an instructor model.

        Args:
            base: input model or model path
            data: instruction-tuning dataset
            task: model task
            prompt: optional prompt template, uses default when not provided
            kwargs: additional training arguments, see HFTrainer docs

        Returns:
            (model, tokenizer)
        """

        # Get prompt
        prompt = prompt if prompt else self.defaultprompt(task)

        # Build training dataset
        train = Dataset.from_generator(self.generate, gen_kwargs=({
            "data": data,
            "task": task,
            "prompt": prompt
            })
        )

        # Train model
        trainer = HFTrainer()
        return trainer(base, train, task=task, **kwargs)

    def generate(self, data, task, prompt):
        """
        Generates an instruction-tuning dataset for training. This method generates fields based on the model task.

        Args:
            data: instruction-tuning dataset
            task: model task
            prompt: input prompt template
        """

        # Template formatter
        formatter = Formatter()

        for row in data:
            for statement in row["statements"]:
                if task == "language-generation":
                    yield {
                        "text": formatter.format(prompt, statement=statement["source"], context=row["context"]) + statement["target"]
                    }
                else:
                    yield {
                        "source": formatter.format(prompt, statement=statement["source"], context=row["context"]),
                        "target": statement["target"]
                    }

    def defaultprompt(self, task):
        """
        Default model prompt.

        Args:
            task: model task

        Returns:
            default model prompt
        """

        template = "Answer the following question using only the context below. Give a detailed answer. "
        template += "Say 'I don't have data on that' when the question can't be answered.\n"
        template += "Question: {statement}\n"
        template += "Context: {context}"
        template += "\nAnswer: " if task == "language-generation" else ""
        return template
