"""A frozen, synthetic corpus for a fictional company, Aurora Cloud.

Hardcoded, not read from disk: instant load, zero drift, and every topic's run.py
imports the exact same 48 documents. Four buckets (hr, product, eng, offtopic) give
retrieval something real to hit and miss. Three near-duplicate pairs (same doc,
one changed number or date) exist so reranking has something concrete to fix.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Doc:
    id: str
    title: str
    tag: str
    text: str
    duplicate_of: str | None = None


DOCS: list[Doc] = [
    # ---- hr (12) --------------------------------------------------------
    Doc(
        "hr-01",
        "Paid Time Off Policy v1",
        "hr",
        "Full-time employees accrue 15 days of paid time off per year. PTO accrues "
        "monthly and can be carried over up to 5 days into the next year. Requests "
        "go through the time-off portal at least two weeks in advance.",
    ),
    Doc(
        "hr-02",
        "Paid Time Off Policy v2",
        "hr",
        "Full-time employees accrue 20 days of paid time off per year. PTO accrues "
        "monthly and can be carried over up to 5 days into the next year. Requests "
        "go through the time-off portal at least two weeks in advance.",
        duplicate_of="hr-01",
    ),
    Doc(
        "hr-03",
        "Remote Work Guidelines",
        "hr",
        "Employees may work remotely up to three days a week with manager approval. "
        "Remote days should be logged in the team calendar. Equipment stipends cover "
        "a monitor and a chair, reimbursed once per year.",
    ),
    Doc(
        "hr-04",
        "Expense Reimbursement Draft A",
        "hr",
        "Expenses under 100 dollars can be submitted without a receipt. Travel "
        "expenses require a manager's pre-approval. Reimbursements are processed "
        "within 10 business days of submission.",
    ),
    Doc(
        "hr-05",
        "Expense Reimbursement Draft B",
        "hr",
        "Expenses under 75 dollars can be submitted without a receipt. Travel "
        "expenses require a manager's pre-approval. Reimbursements are processed "
        "within 10 business days of submission.",
        duplicate_of="hr-04",
    ),
    Doc(
        "hr-06",
        "Parental Leave Policy",
        "hr",
        "New parents receive 16 weeks of paid parental leave, usable within the "
        "first year after birth or adoption. Leave can be split into two blocks of "
        "at least two weeks each.",
    ),
    Doc(
        "hr-07",
        "Code of Conduct",
        "hr",
        "Employees are expected to treat colleagues, customers, and partners with "
        "respect. Harassment of any kind is grounds for immediate termination. "
        "Concerns can be raised anonymously through the ethics hotline.",
    ),
    Doc(
        "hr-08",
        "Onboarding Checklist",
        "hr",
        "New hires complete laptop setup, badge activation, and benefits enrollment "
        "in their first week. A buddy is assigned for the first 30 days to answer "
        "day-to-day questions.",
    ),
    Doc(
        "hr-09",
        "Performance Review Cycle",
        "hr",
        "Performance reviews happen twice a year, in June and December. Each cycle "
        "includes a self-assessment, manager feedback, and calibration across the "
        "team to keep ratings consistent.",
    ),
    Doc(
        "hr-10",
        "Health Insurance Overview",
        "hr",
        "Aurora Cloud covers 90 percent of the premium for the base health plan. "
        "Dental and vision are optional add-ons. Enrollment or changes are allowed "
        "during the open enrollment window each November.",
    ),
    Doc(
        "hr-11",
        "Employee Referral Program",
        "hr",
        "Employees who refer a candidate that gets hired receive a 2000 dollar "
        "bonus, paid out after the new hire's 90-day mark. Referrals are submitted "
        "through the internal careers page.",
    ),
    Doc(
        "hr-12",
        "Sabbatical Policy",
        "hr",
        "Employees with 5 or more years of tenure are eligible for a 4-week paid "
        "sabbatical. Sabbaticals must be scheduled at least 3 months in advance and "
        "approved by a director.",
    ),
    # ---- product (12) -----------------------------------------------------
    Doc(
        "prod-01",
        "Aurora Sync Overview",
        "product",
        "Aurora Sync keeps files up to date across every device automatically. "
        "Changes made on one device appear on others within seconds. Conflicts are "
        "resolved by keeping both versions and flagging them for the user.",
    ),
    Doc(
        "prod-02",
        "Aurora Sync Pricing",
        "product",
        "Aurora Sync's free tier includes 5 gigabytes of storage. The Pro tier is "
        "10 dollars a month for 1 terabyte. Team plans start at 8 dollars per seat "
        "with shared folders included.",
    ),
    Doc(
        "prod-03",
        "Aurora Vault Overview",
        "product",
        "Aurora Vault is the automated backup product. It snapshots a machine every "
        "hour and retains daily snapshots for 30 days. Restoring a file is a single "
        "click from the Vault dashboard.",
    ),
    Doc(
        "prod-04",
        "Aurora Vault Retention Rules",
        "product",
        "Daily snapshots are kept for 30 days, weekly snapshots for 6 months, and "
        "monthly snapshots for 2 years. Deleted files remain recoverable for the "
        "same window as the snapshot that captured them.",
    ),
    Doc(
        "prod-05",
        "Aurora Notify Overview",
        "product",
        "Aurora Notify sends alerts when a monitored system crosses a threshold. "
        "Alerts can go to email, SMS, or a chat webhook. Alert rules support "
        "thresholds, rate-of-change triggers, and quiet hours.",
    ),
    Doc(
        "prod-06",
        "Aurora Notify Escalation Policy",
        "product",
        "An unacknowledged alert escalates to a secondary contact after 10 minutes, "
        "and to the whole on-call team after 30 minutes. Escalation chains are "
        "configured per team in the Notify admin panel.",
    ),
    Doc(
        "prod-07",
        "Aurora Sync Conflict Resolution",
        "product",
        "When two devices edit the same file offline, Aurora Sync keeps both "
        "versions on reconnect, appending the device name to the older one. Users "
        "can merge manually or discard the extra copy.",
    ),
    Doc(
        "prod-08",
        "Aurora Vault Encryption",
        "product",
        "All Aurora Vault snapshots are encrypted at rest with a per-account key. "
        "Keys are rotated every 90 days automatically. Customers on the enterprise "
        "plan can bring their own encryption key.",
    ),
    Doc(
        "prod-09",
        "Aurora Notify Integrations",
        "product",
        "Aurora Notify integrates with Slack, PagerDuty, and generic webhooks. "
        "Setting up an integration takes an API token and a target channel or "
        "endpoint URL.",
    ),
    Doc(
        "prod-10",
        "Aurora Sync Mobile App",
        "product",
        "The Aurora Sync mobile app supports selective sync, so only chosen folders "
        "download to the phone. Photos can auto-upload from the camera roll into a "
        "dedicated synced folder.",
    ),
    Doc(
        "prod-11",
        "Aurora Vault Pricing",
        "product",
        "Aurora Vault costs 6 dollars a month per device for daily backups. "
        "Enterprise plans add custom retention windows and bring-your-own-key "
        "encryption for 15 dollars per device.",
    ),
    Doc(
        "prod-12",
        "Aurora Notify Pricing",
        "product",
        "Aurora Notify's starter plan is free for up to 5 alert rules. Paid plans "
        "start at 20 dollars a month for unlimited rules and multi-channel "
        "escalation.",
    ),
    # ---- eng (12) -----------------------------------------------------------
    Doc(
        "eng-01",
        "On-call Runbook: Database Outage v1",
        "eng",
        "If the primary database is unreachable, page the database on-call "
        "immediately. Check replication lag first, then failover to the standby "
        "if lag exceeds 60 seconds. Document the timeline in the incident channel.",
    ),
    Doc(
        "eng-02",
        "On-call Runbook: Database Outage v2",
        "eng",
        "If the primary database is unreachable, page the database on-call "
        "immediately. Check replication lag first, then failover to the standby "
        "if lag exceeds 30 seconds. Document the timeline in the incident channel.",
        duplicate_of="eng-01",
    ),
    Doc(
        "eng-03",
        "Deploy Process",
        "eng",
        "Deploys go through a staging environment before production. A deploy "
        "requires one approving review and a passing test suite. Production "
        "deploys are limited to business hours unless it's a hotfix.",
    ),
    Doc(
        "eng-04",
        "Incident Severity Levels",
        "eng",
        "Sev1 means the product is down for all customers. Sev2 means a major "
        "feature is broken for some customers. Sev3 covers minor bugs with a "
        "workaround. Sev1 pages the whole on-call rotation immediately.",
    ),
    Doc(
        "eng-05",
        "Postmortem Template",
        "eng",
        "Every Sev1 or Sev2 incident gets a postmortem within 5 business days. "
        "The template covers timeline, root cause, impact, and follow-up action "
        "items with owners and due dates.",
    ),
    Doc(
        "eng-06",
        "Service Level Objectives",
        "eng",
        "The API targets 99.9 percent uptime measured monthly. Latency SLOs are "
        "p50 under 100 milliseconds and p99 under 800 milliseconds for read "
        "endpoints.",
    ),
    Doc(
        "eng-07",
        "Secrets Management",
        "eng",
        "Application secrets are stored in the secrets manager, never in source "
        "control or environment files checked into git. Rotating a secret requires "
        "updating it in the manager and redeploying the affected service.",
    ),
    Doc(
        "eng-08",
        "Code Review Guidelines",
        "eng",
        "Pull requests should be small enough to review in under 20 minutes. "
        "Reviewers check correctness, test coverage, and whether the change matches "
        "existing patterns before approving.",
    ),
    Doc(
        "eng-09",
        "On-call Rotation Schedule",
        "eng",
        "On-call rotates weekly across the platform team. Handoff happens every "
        "Monday at 10am with a short sync covering open issues and anything to "
        "watch for the coming week.",
    ),
    Doc(
        "eng-10",
        "Database Migration Checklist",
        "eng",
        "Migrations run against a staging replica first. Backward-incompatible "
        "changes ship as two-step migrations: add the new column, backfill, then "
        "drop the old one in a later release.",
    ),
    Doc(
        "eng-11",
        "Load Testing Runbook",
        "eng",
        "Load tests run monthly against a staging environment sized to match "
        "production. A test is a pass if p99 latency stays under the SLO target at "
        "twice current peak traffic.",
    ),
    Doc(
        "eng-12",
        "Feature Flag Guidelines",
        "eng",
        "New features ship behind a flag, defaulted off in production. Flags are "
        "removed within one quarter of full rollout to avoid accumulating dead "
        "configuration.",
    ),
    # ---- offtopic (12) --------------------------------------------------
    Doc(
        "off-01",
        "Weeknight Pasta Recipe",
        "offtopic",
        "Boil pasta until just short of al dente. Toast garlic in olive oil, add "
        "chili flakes, then toss the drained pasta in the pan with a splash of "
        "the cooking water to bring the sauce together.",
    ),
    Doc(
        "off-02",
        "History of the Bicycle",
        "offtopic",
        "The earliest bicycles had no pedals and were pushed along with the feet. "
        "Pedals attached directly to the front wheel came decades later, followed "
        "by the chain-driven rear wheel design still used today.",
    ),
    Doc(
        "off-03",
        "Houseplant Watering Tips",
        "offtopic",
        "Most houseplants prefer to dry out between waterings rather than staying "
        "constantly moist. Sticking a finger an inch into the soil is a reliable "
        "way to check before watering again.",
    ),
    Doc(
        "off-04",
        "Board Game Night Ideas",
        "offtopic",
        "Games that support 5 or more players work best for larger groups. "
        "Party games with a fixed round length keep the night moving even if a "
        "few people arrive late.",
    ),
    Doc(
        "off-05",
        "Basics of Sourdough Starter",
        "offtopic",
        "A sourdough starter is just flour and water fermented by wild yeast. "
        "Feeding it daily with fresh flour and water keeps the yeast active enough "
        "to leaven bread.",
    ),
    Doc(
        "off-06",
        "Marathon Training Overview",
        "offtopic",
        "Most marathon plans run 16 to 20 weeks and peak around 40 to 50 miles a "
        "week before a two to three week taper. Long runs build up gradually, "
        "capping around 20 miles.",
    ),
    Doc(
        "off-07",
        "Coffee Brewing Ratios",
        "offtopic",
        "A common starting ratio is 1 gram of coffee to 16 grams of water. Finer "
        "grinds extract faster, so a finer grind needs a shorter brew time to "
        "avoid bitterness.",
    ),
    Doc(
        "off-08",
        "Migratory Bird Patterns",
        "offtopic",
        "Many songbirds migrate at night, using stars and the earth's magnetic "
        "field for orientation. Daytime is spent resting and refueling at stopover "
        "sites along the route.",
    ),
    Doc(
        "off-09",
        "Basic Knife Sharpening",
        "offtopic",
        "A whetstone with a coarse and fine side handles most kitchen knives. "
        "Maintaining a consistent angle, usually 15 to 20 degrees, matters more "
        "than how hard the blade is pressed against the stone.",
    ),
    Doc(
        "off-10",
        "Origins of Chess Openings",
        "offtopic",
        "Many named chess openings date back centuries, refined by generations of "
        "players. The same opening can lead to sharply different games depending "
        "on move order and transpositions.",
    ),
    Doc(
        "off-11",
        "Home Espresso Maintenance",
        "offtopic",
        "Backflushing an espresso machine weekly with a blind basket clears out "
        "coffee oils from the group head. Descaling on a monthly schedule prevents "
        "mineral buildup in the boiler.",
    ),
    Doc(
        "off-12",
        "Basics of Knitting Stitches",
        "offtopic",
        "Knit and purl are the two foundational stitches nearly every pattern "
        "builds on. Alternating them in different patterns produces textures like "
        "ribbing and seed stitch.",
    ),
]

assert len(DOCS) == 48, f"expected 48 docs, got {len(DOCS)}"
assert len({d.id for d in DOCS}) == 48, "doc ids must be unique"
assert sum(1 for d in DOCS if d.duplicate_of) == 3, (
    "expected exactly 3 near-duplicate pairs"
)


if __name__ == "__main__":
    tags = {}
    for d in DOCS:
        tags[d.tag] = tags.get(d.tag, 0) + 1
    print(f"{len(DOCS)} docs: {tags}")
    print("ok")
