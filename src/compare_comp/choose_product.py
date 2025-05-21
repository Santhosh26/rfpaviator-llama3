def chooseCompareProdcucts(selectedOption):
    if selectedOption == "Fortify":
        competativeProdcuts = ["SONARQUBE ", "CHECKMARX ", "SYNPOSYS ", "VERACODE "]
        return competativeProdcuts
    elif selectedOption == "ArcSight":
        competativeProdcuts = ["Splunk ", "Qradar ", "Logrhythm ", "Securonix "]
    elif selectedOption == "NetIQ":
        return

    
    else:
        return None