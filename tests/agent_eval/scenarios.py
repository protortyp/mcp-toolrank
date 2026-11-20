"""Synthetic tasks and tool definitions, intended only for evaluation."""

CATALOGS = {
    "calendar": {
        "lookup": ("Find a calendar event by its title.", "title"),
        "attendees": ("List email addresses of people invited to an event ID.", "event_id"),
        "venue": ("Return a calendar event's meeting room or video location.", "event_id"),
        "cancel_preview": ("Preview notifications if a calendar event were cancelled. Does not cancel.", "event_id"),
        "recurrence": ("Read the recurrence rule of a repeating calendar event.", "event_id"),
        "timezone": ("Look up the timezone for a calendar account.", "account"),
        "list_events": ("List scheduled events for a calendar date.", "date"),
        "availability": ("Find open meeting times on a date.", "date"),
        "lookup_archive": ("Retrieve an archived calendar meeting by its archive reference.", "archive_id"),
        "list_calendars": ("List calendars available to a calendar account.", "account"),
        "find_rooms": ("Find meeting rooms in an office.", "office"),
        "reserve_room_preview": ("Preview a meeting room reservation for a room identifier.", "room"),
    },
    "directory": {
        "lookup": ("Find a staff directory profile by a person's full name.", "name"),
        "office": ("Look up a staff member's office location by their employee ID.", "employee_id"),
        "manager": ("Find the manager of an employee, using an employee ID.", "employee_id"),
        "members": ("List employees in a named department.", "department"),
        "group_alias": ("Look up email address for an internal distribution list.", "group"),
        "public_phone": ("Find an employee's public work phone number.", "employee_id"),
        "personal_phone": ("Find an employee's private personal phone number. Requires HR permission.", "employee_id"),
        "profile_photo": ("Find the avatar for an employee ID.", "employee_id"),
        "schedules": ("Look up a team's business hours for a site.", "site"),
        "list_visitors": ("List registered office visitors for a date.", "date"),
        "employment": ("Check employment status for an employee ID.", "employee_id"),
        "list_departments": ("List departments in a named division.", "division"),
    },
    "support": {
        "lookup": ("Open a support case by its case number and read its current summary.", "case_id"),
        "sla": ("Find the current response deadline for a support case.", "case_id"),
        "ticket_comments": ("Read comments attached to a support case number.", "case_id"),
        "ticket_owner": ("Find the agent assigned to a support case number.", "case_id"),
        "order_lookup": ("Find a retail order by its tracking reference.", "tracking_id"),
        "refund_status": ("Read whether an existing refund was paid for an order.", "order_id"),
        "refund_preview": ("Estimate how much an order would be refunded. Does not issue a refund.", "order_id"),
        "help_article": ("Search published self-service help by phrase.", "phrase"),
        "service_status": ("Read whether an online service is currently operating.", "service"),
        "list_incidents": ("List service incidents for a named product.", "product"),
        "regional_code": ("Find the support escalation code for a region.", "region"),
        "customer_account": ("Read basic account details from a customer number.", "customer_id"),
    },
}
BLOCKED = {"directory/personal_phone"}
RESPONSES = {
    ("calendar/lookup", "Quarterly Demo"): "Calendar event E42, Quarterly Demo.",
    ("calendar/lookup", "Old Roadmap"): "Archived. The archive tool is now available. Archive reference A7.",
    ("calendar/lookup_archive", "A7"): "The old roadmap meeting was on March 9.",
    ("directory/lookup", "Riley Chen"): "Employee ID P17, Riley Chen.",
    ("support/lookup", "C12"): "Case C12 is urgent. Follow up by getting the escalation code for region West.",
    ("support/regional_code", "West"): "Escalation code W-4.",
}

# The requested steps use the real argument names exposed by each schema.
CASES = [
    ("calendar-title", "Find the calendar event titled Quarterly Demo.", [("calendar/lookup", "Quarterly Demo")]),
    ("directory-name", "Look up the staff directory profile for Riley Chen.", [("directory/lookup", "Riley Chen")]),
    ("same-name", "Open support case C12 and show its summary only.", [("support/lookup", "C12")]),
    ("availability", "Find a free meeting slot on 2026-12-10.", [("calendar/availability", "2026-12-10")]),
    ("similar-tools", "Check whether the existing refund for order O8 was paid.", [("support/refund_status", "O8")]),
    ("authorization", "Get the public work phone for employee P17. Use the public directory.", [("directory/public_phone", "P17")]),
    ("two-independent", "Find the meeting location and attendees for event E42.",
     [("calendar/venue", "E42"), ("calendar/attendees", "E42")]),
    ("dependent", "Look up support case C12, then follow its instruction for finding the required code.",
     [("support/lookup", "C12"), ("support/regional_code", "West")]),
    ("refresh", "Look up the meeting Old Roadmap. If archived, retrieve the original date.",
     [("calendar/lookup", "Old Roadmap"), ("calendar/lookup_archive", "A7")]),
    ("absent", "Translate the phrase buenos días into Swedish.", []),
    ("denied", "Get employee P17's private personal phone number.", []),
]


def schema(argument):
    return {"type": "object", "properties": {argument: {"type": "string"}},
            "required": [argument], "additionalProperties": False}
