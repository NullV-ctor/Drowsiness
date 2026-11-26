"""Daily episode reports contain only date, time slots, and total count."""

import csv
import io
from datetime import datetime
from zoneinfo import ZoneInfo


def report_rows(episodes, date, timezone="Asia/Kolkata"):
    zone = ZoneInfo(timezone)
    rows = []
    for episode in episodes:
        start = datetime.fromisoformat(episode["started_at"]).astimezone(zone)
        if start.date() != date:
            continue  # Cross-midnight episodes belong to their start date and are not double-counted.
        end = (
            datetime.fromisoformat(episode["ended_at"]).astimezone(zone)
            if episode["ended_at"]
            else None
        )
        rows.append(
            {
                "Start": start.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "End": end.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                if end
                else "Ongoing / end not observed",
            }
        )
    return sorted(rows, key=lambda row: row["Start"])


def report_csv(date, rows):
    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(["Date", "Start", "End", "Total episodes"])
    for row in rows:
        writer.writerow([date.isoformat(), row["Start"], row["End"], len(rows)])
    if not rows:
        writer.writerow([date.isoformat(), "", "", 0])
    return text.getvalue().encode()


def report_pdf(date, rows):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )

    output = io.BytesIO()
    doc = SimpleDocTemplate(
        output,
        pagesize=A4,
        leftMargin=45,
        rightMargin=45,
        topMargin=45,
        bottomMargin=45,
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph(f"Date: {date.isoformat()}", styles["Normal"]),
        Spacer(1, 12),
        Paragraph(f"Total episodes: {len(rows)}", styles["Normal"]),
        Spacer(1, 18),
    ]
    if rows:
        data = [["Start", "End"]] + [[row["Start"], row["End"]] for row in rows]
        table = Table(data, colWidths=[245, 245], repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.lightgrey),
                ]
            )
        )
        story.append(table)
    doc.build(story)
    return output.getvalue()
