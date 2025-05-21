def status_checker(result):
    # Check if the result starts with "Yes" to determine if it's positive
    if result.strip().startswith("Yes"):
        return "Compliant"
    elif result.strip().startswith("No"):
        return "Non-Compliant"
    else:
        return "Review required"