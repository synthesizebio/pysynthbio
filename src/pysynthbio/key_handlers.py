import getpass
import os
import warnings
import webbrowser

try:
    import keyring

    KEYRING_AVAILABLE = True
except Exception:
    KEYRING_AVAILABLE = False
    warnings.warn(
        "Failed to import 'keyring' `use_keyring` will fail:"
        "\n Do pip install keyring if you'd like this feature \n",
        stacklevel=2,
    )


def set_synthesize_token(use_keyring=False, token=None):
    """
    Securely prompts for and stores the Synthesize Bio API
    token in the environment.

    Args:
        use_keyring (bool): Whether to also store the token
            securely in the system keyring for future sessions.
            Defaults to False.
        token (str, optional): If provided, uses this token
            instead of prompting. This parameter should only
            be used in non-interactive scripts.

    Returns:
        bool: True if successful.

    Examples:
        # Interactive prompt for token
        set_synthesize_token()

        # Provide token directly (less secure, not recommended for interactive use)
        set_synthesize_token(token="your-token-here")

        # Store in system keyring for future sessions
        set_synthesize_token(use_keyring=True)
    """

    if token is None:
        webbrowser.open("https://app.synthesize.bio/account/api-keys")
        token = getpass.getpass(
            prompt="Create an account at https://app.synthesize.bio/ \n"
            "Go to your account page.\n"
            "Click create API key then copy it.\n"
            "Paste token here and press enter: "
        )

    # Store in environment
    os.environ["SYNTHESIZE_API_KEY"] = token

    # Optionally store in keyring if requested and available
    if use_keyring:
        if KEYRING_AVAILABLE:
            try:
                keyring.set_password("pysynthbio", "api_token", token)
                print("API token stored in system keyring.")
            except Exception as e:
                warnings.warn(
                    f"Failed to store token in keyring: {str(e)}", stacklevel=2
                )
        else:
            warnings.warn(
                "Package 'keyring' is not installed. Token not stored in keyring.",
                stacklevel=2,
            )
            print("To store token in keyring, install with: pip install keyring")

    print("API token set for current session.")
    return True


def load_synthesize_token_from_keyring():
    """
    Loads the previously stored Synthesize Bio API token from the system
    keyring and sets it in the environment for the current session.

    Returns:
        bool: True if successful, False if token not found in keyring.

    Examples:
        # Load token from keyring
        load_synthesize_token_from_keyring()
    """
    if not KEYRING_AVAILABLE:
        warnings.warn(
            "Package 'keyring' is not installed.",
            "Cannot load token from keyring.",
            stacklevel=2,
        )
        print("To use this feature, install with: pip install keyring")
        return False

    try:
        token = keyring.get_password("pysynthbio", "api_token")
        if token is None:
            warnings.warn("No token found in keyring.", stacklevel=2)
            return False

        os.environ["SYNTHESIZE_API_KEY"] = token
        print("API token loaded from keyring and set for current session.")
        return True
    except Exception as e:
        warnings.warn(f"Failed to load token from keyring: {str(e)}", stacklevel=2)
        return False


def clear_synthesize_token(remove_from_keyring=False):
    """
    Clears the Synthesize Bio API token from the environment for the
    current Python session. This is useful for security purposes when you've finished
    working with the API or when switching between different accounts.

    Args:
        remove_from_keyring (bool): Whether to also remove the token from the
            system keyring if it's stored there. Defaults to False.

    Returns:
        bool: True

    Examples:
        # Clear token from current session only
        clear_synthesize_token()

        # Clear token from both session and keyring
        clear_synthesize_token(remove_from_keyring=True)
    """
    # Unset from environment
    if "SYNTHESIZE_API_KEY" in os.environ:
        del os.environ["SYNTHESIZE_API_KEY"]
        print("API token cleared from current session.")
    else:
        print("No API token was set in the current session.")

    # Optionally remove from keyring
    if remove_from_keyring:
        if KEYRING_AVAILABLE:
            try:
                keyring.delete_password("pysynthbio", "api_token")
                print("API token removed from system keyring.")
            except keyring.errors.PasswordDeleteError:
                print("No API token was found in the keyring.")
            except Exception:
                # This might occur if no token exists or other keyring issues
                print(
                    "No API token was found in the keyring",
                    " or could not access keyring.",
                )
        else:
            warnings.warn(
                "Package 'keyring' is not installed.",
                "Cannot remove token from keyring.",
                stacklevel=2,
            )
    print("To use this feature, install with: pip install keyring")

    return True


def has_synthesize_token():
    """
    Checks whether a Synthesize Bio API token is currently set in the
    environment. Useful for conditional code that requires an API token.

    Returns:
        bool: True if token is set, False otherwise.

    Examples:
        # Check if token is set
        if not has_synthesize_token():
            # Prompt for token if not set
            set_synthesize_token()
    """
    return "SYNTHESIZE_API_KEY" in os.environ and os.environ["SYNTHESIZE_API_KEY"] != ""
