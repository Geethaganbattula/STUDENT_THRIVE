from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Define stress level ranges
stress_levels = {
    0: "Low",
    1: "Medium",
    2: "High"
}

# Model Loading (with error handling)
try:
    with open('stress_model.pkl', 'rb') as f:
        tree_clf = pickle.load(f)
except FileNotFoundError:
    try:
        h = pd.read_csv("StressLevelDataset.csv")
        h.dropna(inplace=True)
        X = h.drop("stress_level", axis=1)
        y = h["stress_level"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        tree_clf = DecisionTreeClassifier(max_depth=7, random_state=42)
        tree_clf.fit(X_train, y_train)

        with open('stress_model.pkl', 'wb') as f:
            pickle.dump(tree_clf, f)
        print("Model trained and saved to stress_model.pkl")

    except FileNotFoundError:
        print("Error: StressLevelDataset.csv not found.")
        exit()
    except Exception as e:
        print(f"An error occurred during model training: {e}")
        exit()

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET'])
def login():
    return render_template('login.html')

@app.route('/register', methods=['GET'])
def register():
    return render_template('register.html')

@app.route('/survey', methods=['GET', 'POST'])
def survey():
    if request.method == 'POST':
        try:
            angry_level = float(request.form['Angry_level'])
            situation_understand_ability = float(request.form['situation_understand_ability'])
            nervousness = float(request.form['nervousness'])
            headache = float(request.form['headache'])
            sleep_quality = float(request.form['sleep_quality'])
            breathing_problem = float(request.form['breathing_problem'])
            living_conditions = float(request.form['living_conditions'])
            academic_performance = float(request.form['academic_performance'])
            study_load = float(request.form['study_load'])
            future_career_concerns = float(request.form['future_career_concerns'])
            extracurricular_activities = float(request.form['extracurricular_activities'])

            user_input = pd.DataFrame([[angry_level, situation_understand_ability, nervousness, headache,
                                        sleep_quality, breathing_problem, living_conditions,
                                        academic_performance, study_load, future_career_concerns,
                                        extracurricular_activities]],
                                     columns=['anxiety_level', 'mental_health_history', 'depression', 'headache',
                                              'sleep_quality', 'breathing_problem', 'living_conditions',
                                              'academic_performance', 'study_load', 'future_career_concerns',
                                              'extracurricular_activities'])

            predicted_stress_level_int = tree_clf.predict(user_input)[0]
            predicted_stress_level = stress_levels.get(predicted_stress_level_int, "Unknown")
            advices = get_advices(predicted_stress_level_int)

            if predicted_stress_level_int == 0:
                stress_class = "low"
            elif predicted_stress_level_int == 1:
                stress_class = "medium"
            elif predicted_stress_level_int == 2:
                stress_class = "high"
            else:
                stress_class = ""

            return render_template('result.html', predicted_stress_level=predicted_stress_level, advices=advices, stress_class=stress_class)

        except ValueError:
            return "Invalid input. Please enter numbers only."
        except Exception as e:
            return f"An error occurred: {e}"

    return render_template('survey.html')


def get_advices(predicted_stress_level_int):
    if predicted_stress_level_int == 0:
        return [
            "ADVICE:Maintain a Balanced Routine",
            "* Practice Time Management",
            "* Stay Organized",
            "* Set Realistic Goals",
            "* Stay Active",
            "* Practice Relaxation Techniques",
            "* Seek Help When Needed"
        ]
    elif predicted_stress_level_int == 1:
        return [
            "* Practice Mindfulness",
            "* Journaling: Gratitude Practice",
            "* Creative Outlets",
            "* Nature Time",
            "* Limit Screen Time",
            "* Practice Assertiveness"
        ]
    elif predicted_stress_level_int == 2:
        return [
            "* Floating Therapy",
            "* Forest Bathing",
            "* Art Therapy",
            "* Laughter Yoga",
            "* Equine-Assisted Therapy",
            "* Drum Circles",
            "* Aerial Yoga",
            "* Color Therapy",
            "* Sensory Deprivation Tank",
            "* Urban Foraging"
        ]
    else:
        return ["Invalid stress level."]

if __name__ == '__main__':
    app.run(debug=True)