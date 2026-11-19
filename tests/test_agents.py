from civintel_agentbench.agents import cross_source_analyst_agent, evidence_auditor_agent


def test_auditor_abstains_instead_of_claiming_truth():
    pair = {
        "left_report_id": "a", "right_report_id": "b", "similarity": 0.9,
        "same_source_family": False, "left_source": "Source A", "right_source": "Source B",
    }
    analysed = cross_source_analyst_agent([pair])
    audited = evidence_auditor_agent(analysed)
    assert audited[0]["decision"] == "abstain"
    assert audited[0]["confidence"] == 0.0
    assert "not proof" in audited[0]["audit_warning"]
