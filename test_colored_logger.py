"""
Logger Color Test

This script demonstrates the colored logging functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypersurrogatemodel import Logger


def test_colored_logger():
    """
    Test the colored logger functionality.
    """
    print("=== Testing Colored Logger ===\n")
    
    # Create colored logger
    logger = Logger("color_test", use_colors=True)
    
    print("Colored Logger Output:")
    logger.debug("This is a DEBUG message (Cyan)")
    logger.info("This is an INFO message (Green)")
    logger.warning("This is a WARNING message (Yellow)")
    logger.error("This is an ERROR message (Red)")
    logger.critical("This is a CRITICAL message (Magenta)")
    
    print("\nSpecial formatted messages:")
    logger.success("This is a SUCCESS message with checkmark")
    logger.step("This is a STEP message with spinner icon")
    logger.result("This is a RESULT message with chart icon")
    
    print("\n" + "="*50)
    
    # Create non-colored logger for comparison
    logger_no_color = Logger("no_color_test", use_colors=False)
    
    print("\nNon-Colored Logger Output:")
    logger_no_color.debug("This is a DEBUG message (no color)")
    logger_no_color.info("This is an INFO message (no color)")
    logger_no_color.warning("This is a WARNING message (no color)")
    logger_no_color.error("This is an ERROR message (no color)")
    logger_no_color.critical("This is a CRITICAL message (no color)")


if __name__ == "__main__":
    test_colored_logger()
