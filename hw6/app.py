"""A simple command-line password strength checker.

This program only evaluates the password while it is in memory.  It does not
save, log, or transmit the password.
"""

import getpass
import string

MAX_PASSWORD_LENGTH = 1024


def check_password(password):
    """Return a strength rating, explanations, and improvement suggestions."""
    # Avoid doing unnecessary work on an accidentally or deliberately huge
    # input.  The password itself is never included in the returned messages.
    if not isinstance(password, str) or len(password) > MAX_PASSWORD_LENGTH:
        return (
            "Not evaluated",
            [f"The password is longer than the {MAX_PASSWORD_LENGTH}-character limit."],
            ["Choose a shorter password or passphrase."],
        )

    has_uppercase = any(character.isupper() for character in password)
    has_lowercase = any(character.islower() for character in password)
    has_number = any(character.isdigit() for character in password)
    has_special = any(character in string.punctuation for character in password)
    has_whitespace = any(character.isspace() for character in password)
    has_control_character = any(
        not character.isprintable() for character in password
    )
    long_enough = len(password) >= 12

    # Each satisfied requirement contributes one point to the basic estimate.
    checks = [long_enough, has_uppercase, has_lowercase, has_number, has_special]
    score = sum(checks)

    if len(password) < 8 or score <= 2:
        rating = "Weak"
    elif score == 3 or score == 4:
        rating = "Moderate"
    else:
        rating = "Strong"

    reasons = [
        f"Length: {len(password)} characters (12 or more is recommended).",
        "Contains uppercase letters." if has_uppercase
        else "Does not contain uppercase letters.",
        "Contains lowercase letters." if has_lowercase
        else "Does not contain lowercase letters.",
        "Contains numbers." if has_number else "Does not contain numbers.",
        "Contains special characters." if has_special
        else "Does not contain special characters.",
    ]

    # Whitespace is allowed as part of a password, but is not treated as an
    # ASCII special character.  Control characters can be difficult to enter
    # consistently and may be handled differently by other systems.
    if has_whitespace:
        reasons.append("Contains whitespace; whitespace is not counted as a special character.")
    if has_control_character:
        reasons.append("Contains a non-printable character.")

    suggestions = []
    if len(password) < 12:
        suggestions.append("Use at least 12 characters; a longer passphrase is even better.")
    if not has_uppercase:
        suggestions.append("Add one or more uppercase letters.")
    if not has_lowercase:
        suggestions.append("Add one or more lowercase letters.")
    if not has_number:
        suggestions.append("Add one or more numbers.")
    if not has_special:
        suggestions.append("Add a special character such as !, @, or #.")
    if has_control_character:
        suggestions.append("Avoid non-printable control characters for compatibility.")

    return rating, reasons, suggestions


def main():
    """Read one password and display its evaluation without displaying it."""
    print("Password Strength Checker")
    print("Your password is evaluated locally and is not saved or transmitted.")

    try:
        password = getpass.getpass("Enter a password to check: ")
    except (EOFError, KeyboardInterrupt, OSError):
        print("\nNo password was entered. Exiting.")
        return

    if not password:
        print("No password was entered. Please enter at least one character.")
        return

    rating, reasons, suggestions = check_password(password)

    print(f"\nStrength rating: {rating}")
    print("Why:")
    for reason in reasons:
        print(f"- {reason}")

    if suggestions:
        print("Suggestions:")
        for suggestion in suggestions:
            print(f"- {suggestion}")

    print("\nThis is a basic estimate and does not guarantee that a password is secure.")


if __name__ == "__main__":
    main()
