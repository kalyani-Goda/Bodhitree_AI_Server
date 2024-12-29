import json


def extract_data():
    """Function to extracting the data from the Database"""
    # Fetch all Problem instances 
    all_problems = Problem.objects.all()
    # Choose problem where isTrained is False
    untrained_problems = [problem for problem in all_problems if not problem.isTrained]
    # print(f"Untrained Problems: {untrained_problems}")
    totalDataPoints = 0

    # Dictionary to store the final JSON structure
    json_data = {}

    for problem in untrained_problems:
        totalDataPoints += problem.totalDataPoints
        # Fetch The Problem Statement
        try:
            # problem_id = int(input("Enter the ID of the problem you want to see: "))
            problem_id = problem.id  
            problem = Problem.objects.get(id=problem_id)
            problemStatement = problem.problem_statement
        except Problem.DoesNotExist:
            print(f"No problem found with ID: {problem_id}")
            return

        # Fetch all Submissions for the given problem
        submissions = Submission.objects.filter(problem_id=problem_id)
        studentSubmissions = {submission.student_id: submission.source_code for submission in submissions}

        # Fetch the Rubrics for the given problem
        rubrics = {}
        try:
            # Fetch all criteria for the given problem, including their associated ratings
            criteria_with_ratings = Criteria.objects.filter(problem_id=problem_id).prefetch_related('rating_set')
            
            for criteria in criteria_with_ratings:
                criteriaDict = {'description': criteria.description}
                ratingsDict = {rating.title: rating.description for rating in criteria.rating_set.all()}
                if not ratingsDict:
                    print("  No ratings available for this criteria.")
                criteriaDict['ratings'] = ratingsDict
                rubrics[criteria.title] = criteriaDict
        except Problem.DoesNotExist:
            print(f"No problem found with ID: {problem_id}")
        
        # Fetch the Original Grades for the given problem
        grades = {}
        submissions = Submission.objects.filter(problem_id=problem_id)

        for submission in submissions:
            grading_histories = GradingHistory.objects.filter(submission=submission.id)
            
            for grading_history in grading_histories:
                criterion_title = grading_history.criteria.title
                student_id = submission.student_id
                try:
                    manual_marks = grading_history.manual_rating.title
                except GradingHistory.manual_rating.RelatedObjectDoesNotExist:
                    print("Manual Rating: Not present")
                    continue
                
                # Initialize the nested dictionary for each criterion title if it doesn't exist
                if criterion_title not in grades:
                    grades[criterion_title] = {}

                # Store the student_id as the key and manual_rating.marks as the value
                grades[criterion_title][student_id] = manual_marks
        # Populate the JSON dictionary for the current problem
        json_data[problem_id] = {
            'problem_statement': problemStatement,
            'student_submissions': studentSubmissions,
            'rubrics': rubrics,
            'grades': grades,
        }
    return json.dumps(json_data), totalDataPoints