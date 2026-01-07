import re
from datetime import datetime, timedelta


class NoteParser:
    """Parser for structured medical notes with date and type information."""
    
    DEFAULT_SEPARATOR = r"<NOTE:([\s\S]+?)(\d{4}-\d{2}-\d{2})>"
    DEFAULT_KEYS = ["note_type", "date", "note"]
    DATE_FORMAT = "%Y-%m-%d"
    
    def __init__(self, separator: str = DEFAULT_SEPARATOR, keys: list[str] = DEFAULT_KEYS):
        self.separator = re.compile(separator)
        self.keys = keys
    
    def parse(self, notes: str) -> list[dict]:
        """Parse notes string into structured dictionaries."""
        parts = self.separator.split(notes)
        
        # Calculate capture groups from separator pattern
        capture_count = self.separator.groups
        
        parsed_notes = []
        for i in range(1, len(parts), capture_count + 1):
            if i + capture_count < len(parts):
                note_data = tuple(parts[i + j].strip() for j in range(capture_count + 1))
                parsed_notes.append(self._to_dict(note_data))
        
        return parsed_notes
    
    def _to_dict(self, note_tuple: tuple) -> dict:
        """Convert note tuple to dictionary with validation."""
        if len(note_tuple) != len(self.keys):
            raise ValueError(
                f"Note tuple length ({len(note_tuple)}) does not match "
                f"number of keys ({len(self.keys)})"
            )
        return dict(zip(self.keys, note_tuple))
    
    def _parse_date(self, date_str: str, date_format: str | None = DATE_FORMAT) -> datetime.date: # type: ignore
        """Parse date string to date object."""
        date_format = date_format or self.DATE_FORMAT
        return datetime.strptime(date_str, date_format).date()
    
    @staticmethod
    def format_note(note_type: str, date: str, content: str, **_) -> str:
        """Format note with standard header."""
        return f"<NOTE: {note_type} {date}> {content}"


class NoteFilter:
    """Filter notes based on date ranges and note types."""
    
    def __init__(
        self,
        note_types_to_keep: str | list[str] | None = None,
        days_before: int | None = None,
        days_after: int | None = None,
        reference_date_format: str | None = None,
        note_date_format: str | None = None,
        parser: NoteParser | None = None
    ):
        self.note_types_to_keep = [note_types_to_keep] if isinstance(note_types_to_keep, str) else note_types_to_keep
        self.days_before = days_before
        self.days_after = days_after
        self.reference_date_format = reference_date_format
        self.note_date_format = note_date_format
        self.parser = parser or NoteParser()

    @staticmethod
    def _in_date_range(
        reference_date: datetime.date | None = None, # type: ignore
        start_date: datetime.date | None = None, # type: ignore
        end_date: datetime.date | None = None # type: ignore
    ) -> bool:
        if (reference_date is None) or (start_date is None and end_date is None):
            return True
        
        after_start = reference_date >= start_date if start_date is not None else True
        before_end = reference_date <= end_date if end_date is not None else True

        return after_start and before_end
    
    def in_date_range(
        self,
        parsed_note: dict,
        reference_date: str,
        days_before: int | None,
        days_after: int | None,
        note_date_format: str | None = None,
        reference_date_format: str | None = None
    ) -> bool:
        note_date = self.parser._parse_date(parsed_note["date"], note_date_format)
        reference_date = self.parser._parse_date(reference_date, reference_date_format)
        start_date = reference_date - timedelta(days=days_before) if days_before is not None else None # type: ignore
        end_date = reference_date + timedelta(days=days_after) if days_after is not None else None # type: ignore
        return self._in_date_range(note_date, start_date, end_date)
    
    def filter_by_date_range(
        self,
        notes: str,
        reference_date: str,
        reference_date_format: str | None = None,
        note_date_format: str | None = None,
        days_before: int | None = None,
        days_after: int | None = None,
        format_string: bool = True
    ) -> str | list[dict]:
        """Filter notes within a date range around a reference date."""
        date_kwargs = {
            "reference_date": reference_date,
            "days_before": days_before or self.days_before,
            "days_after": days_after or self.days_after,
            "note_date_format": note_date_format or self.note_date_format,
            "reference_date_format": reference_date_format or self.reference_date_format
        }
        
        if date_kwargs["days_before"] is None:
            raise ValueError("No 'days_before' value found")
        if date_kwargs["days_after"] is None:
            raise ValueError("No 'days_after' value found")

        return self._apply_filter(notes=notes, **date_kwargs)
    
    @staticmethod
    def _keep_note_type(
        note_type: str | dict,
        note_types_to_keep: list[str] | None = None
    ) -> bool:
        if note_types_to_keep is None:
            return True
        
        if isinstance(note_type, dict):
            note_type = note_type.get("note_type", "")
        
        return note_type.lower() in [keep_type.lower() for keep_type in note_types_to_keep] # type: ignore
    
    def filter_by_note_types(
        self,
        notes: str,
        note_types: list[str] | None = None,
        format_string: bool = True
    ) -> str | list[dict]:
        """Filter notes by allowed note types."""
        note_types = note_types or self.note_types_to_keep

        return self._apply_filter(notes=notes, note_types=note_types, format_string=format_string)
    
    def _apply_filter(
        self,
        notes: str,
        reference_date: str | None = None,
        reference_date_format: str | None = None,
        note_date_format: str | None = None,
        note_types: list[str] | None = None,
        days_before: int | None = None,
        days_after: int | None = None,
        format_string: bool = True
    ) -> str | list[dict]:
        """Apply filters and return formatted notes string."""
        if (note_types is None) and (days_before is None) and (days_after is None):
            raise ValueError("No filter parameters found")
        
        date_kwargs = {
            "reference_date": reference_date,
            "days_before": days_before,
            "days_after": days_after,
            "note_date_format": note_date_format,
            "reference_date_format": reference_date_format
        }
        parsed_notes = [note for note in self.parser.parse(notes) if self._keep_note_type(note, note_types) and self.in_date_range(note, **date_kwargs)]
        
        if format_string:
            return " ".join([self.parser.format_note(**note) for note in parsed_notes])
        
        return parsed_notes

    
    def apply_filters(self, notes: str, reference_date: str | None = None, format_string: bool = True) -> str | list[dict]:
        """Apply filters """
        return self._apply_filter(
            notes=notes,
            reference_date=reference_date,
            note_types=self.note_types_to_keep,
            reference_date_format=self.reference_date_format,
            note_date_format=self.note_date_format,
            days_before=self.days_before,
            days_after=self.days_after,
            format_string=format_string
        )
