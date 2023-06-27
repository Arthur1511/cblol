from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_model_input_table, preprocess_teams


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_teams,
                inputs="teams",
                outputs="preprocessed_teams",
                name="preprocess_teams_node",
            ),
            node(
                func=create_model_input_table,
                inputs=["preprocessed_teams"],
                outputs="model_input_table",
                name="create_model_input_table_node",
            ),
        ]
    )
