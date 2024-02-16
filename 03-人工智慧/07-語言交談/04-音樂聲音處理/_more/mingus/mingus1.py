import mingus.core.notes as notes

print(notes.is_valid_note("C"))
print(notes.is_valid_note("C######bb"))
print(notes.remove_redundant_accidentals("C##b"))
print(notes.note_to_int("C"))
print(notes.note_to_int("B"))
print(notes.int_to_note(11))