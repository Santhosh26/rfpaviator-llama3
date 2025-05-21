def productCompalincePercentage(status_list):
    complaint_score = 0
    noncomplaint_score = 0

    for status in status_list:
        if status == "Compliant":
            complaint_score = complaint_score + 1
        else:
            noncomplaint_score = noncomplaint_score + 1
    total_score = complaint_score + noncomplaint_score



    comp_precentage =  (complaint_score / total_score)* 100
    noncomp_precentage =  (noncomplaint_score / total_score)* 100
    return (round(comp_precentage), round(noncomp_precentage))


# slist = ["Complaint", "non-complaint", "Complaint", "Complaint", "Complaint", " "]
# comp_perlist = productCompalincePercentage(status_list=slist)

# print(f"complaint percentage: {comp_perlist[0]}")
# print(f"non complaint percentage: {comp_perlist[1]}")

    

