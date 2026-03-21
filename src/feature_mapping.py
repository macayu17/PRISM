class FeatureMapper:
    def __init__(self):
        # Mapping between patient-facing questions and the trained inference schema.
        self.feature_mapping = {
            "basic_info": {
                "age": {
                    "question": "What is your age?",
                    "type": "numeric",
                    "dataset_feature": "age",
                },
                "sex": {
                    "question": "What is your biological sex?",
                    "type": "categorical",
                    "options": ["Male", "Female"],
                    "dataset_feature": "SEX",
                    "mapping": {"Male": 1, "Female": 0},
                },
                "education": {
                    "question": "Years of education completed?",
                    "type": "numeric",
                    "dataset_feature": "EDUCYRS",
                },
                "race": {
                    "question": "What is your race?",
                    "type": "categorical",
                    "options": ["White", "Black/African American", "Asian", "Other"],
                    "dataset_feature": "race",
                    "mapping": {
                        "White": 1,
                        "Black/African American": 2,
                        "Asian": 3,
                        "Other": 4,
                    },
                },
                "bmi": {
                    "question": "What is your BMI?",
                    "type": "numeric",
                    "dataset_feature": "BMI",
                },
            },
            "family_history": {
                "family_pd": {
                    "question": "Do you have any family members with Parkinson's disease?",
                    "type": "categorical",
                    "options": [
                        "No family history",
                        "First degree relative",
                        "Other relative",
                    ],
                    "dataset_feature": "fampd",
                    "mapping": {
                        "No family history": 3,
                        "First degree relative": 1,
                        "Other relative": 2,
                    },
                }
            },
            "motor_symptoms": {
                "tremor": {
                    "question": "Tremor severity (0-4)",
                    "type": "numeric",
                    "dataset_feature": "sym_tremor",
                    "scale": "0-4",
                },
                "rigidity": {
                    "question": "Rigidity severity (0-4)",
                    "type": "numeric",
                    "dataset_feature": "sym_rigid",
                    "scale": "0-4",
                },
                "bradykinesia": {
                    "question": "Bradykinesia severity (0-4)",
                    "type": "numeric",
                    "dataset_feature": "sym_brady",
                    "scale": "0-4",
                },
                "balance": {
                    "question": "Postural instability severity (0-4)",
                    "type": "numeric",
                    "dataset_feature": "sym_posins",
                    "scale": "0-4",
                },
            },
            "non_motor_symptoms": {
                "rem_sleep": {
                    "question": "Do you act out dreams or have REM sleep behaviour symptoms?",
                    "type": "categorical",
                    "options": ["No", "Yes"],
                    "dataset_feature": "rem",
                    "mapping": {"No": 0, "Yes": 1},
                },
                "daytime_sleepiness": {
                    "question": "Epworth Sleepiness Scale score?",
                    "type": "numeric",
                    "dataset_feature": "ess",
                    "scale": "0-24",
                },
                "depression": {
                    "question": "Geriatric Depression Scale score?",
                    "type": "numeric",
                    "dataset_feature": "gds",
                    "scale": "0-15",
                },
                "anxiety": {
                    "question": "State-Trait Anxiety Inventory score?",
                    "type": "numeric",
                    "dataset_feature": "stai",
                    "scale": "20-80",
                },
            },
            "cognitive_symptoms": {
                "memory": {
                    "question": "MoCA score?",
                    "type": "numeric",
                    "dataset_feature": "moca",
                    "scale": "0-30",
                },
                "clock_draw": {
                    "question": "Clock drawing test score?",
                    "type": "numeric",
                    "dataset_feature": "clockdraw",
                    "scale": "0-4",
                },
                "bjlot": {
                    "question": "Benton line orientation score?",
                    "type": "numeric",
                    "dataset_feature": "bjlot",
                    "scale": "0-30",
                },
            },
        }

    def get_patient_questionnaire(self):
        """Generate a list of questions for patients."""
        questions = []
        for category in self.feature_mapping.values():
            for feature in category.values():
                questions.append(
                    {
                        "question": feature["question"],
                        "type": feature["type"],
                        "options": feature.get("options"),
                        "scale": feature.get("scale"),
                    }
                )
        return questions

    def map_patient_response_to_features(self, responses):
        """Map patient responses to dataset features."""
        feature_values = {}
        for category in self.feature_mapping.values():
            for feature_name, feature_info in category.items():
                if feature_name not in responses:
                    continue

                response = responses[feature_name]
                dataset_feature = feature_info["dataset_feature"]

                if "mapping" in feature_info:
                    mapped_value = feature_info["mapping"].get(response)
                else:
                    mapped_value = response

                if mapped_value is None:
                    continue

                feature_values[dataset_feature] = mapped_value

                if dataset_feature == "fampd":
                    feature_values["fampd_bin"] = 2 if mapped_value == 3 else 1

        return feature_values


def main():
    mapper = FeatureMapper()
    questions = mapper.get_patient_questionnaire()

    print("Patient Questionnaire:")
    for i, q in enumerate(questions, 1):
        print(f"\n{i}. {q['question']}")
        if q["options"]:
            print(f"Options: {', '.join(q['options'])}")
        if q["scale"]:
            print(f"Scale: {q['scale']}")


if __name__ == "__main__":
    main()
