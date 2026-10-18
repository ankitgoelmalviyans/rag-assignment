# Conftest / OPA policy run against `terraform show -json` output.
# This is the HARD gate: AI-written infra must pass the same rules as human-written infra.
package main

import rego.v1

required_tags := {"owner", "classification", "cost_center"}

changes contains rc if {
	some rc in input.resource_changes
	rc.change.actions[_] != "no-op"
}

deny contains msg if {
	some rc in changes
	rc.type == "aws_s3_bucket"
	missing := required_tags - {k | some k, _ in rc.change.after.tags}
	count(missing) > 0
	msg := sprintf("%s is missing required tags: %v", [rc.address, missing])
}

deny contains msg if {
	some rc in changes
	rc.type == "aws_s3_bucket_public_access_block"
	some flag in ["block_public_acls", "block_public_policy", "ignore_public_acls", "restrict_public_buckets"]
	rc.change.after[flag] != true
	msg := sprintf("%s must set %s = true", [rc.address, flag])
}

deny contains msg if {
	some rc in changes
	rc.type == "aws_iam_policy"
	doc := json.unmarshal(rc.change.after.policy)
	some stmt in doc.Statement
	stmt.Effect == "Allow"
	stmt.Action == "*"
	msg := sprintf("%s grants Action '*' (wildcard IAM is not allowed)", [rc.address])
}

deny contains msg if {
	some rc in changes
	"delete" in rc.change.actions
	rc.change.before.tags.classification in {"confidential", "restricted"}
	msg := sprintf("%s: deleting a %s resource needs a manual change ticket", [rc.address, rc.change.before.tags.classification])
}
