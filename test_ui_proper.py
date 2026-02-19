"""
Proper UI Test for J.A.R.V.I.S. Trip Booking - WITH RESULT VERIFICATION
Tests the UI interaction and verifies command results are displayed
"""

import asyncio
from playwright.async_api import async_playwright
import os
import time

async def test_trip_booking_ui_with_results():
    """Test the UI trip booking functionality with result verification"""

    async with async_playwright() as p:
        # Launch browser in headless mode for faster testing
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1280, 'height': 720})
        page = await context.new_page()

        try:
            print("📸 Step 1: Navigating to J.A.R.V.I.S. UI...")

            # Navigate to UI
            await page.goto("http://localhost:5174/J.A.R.V.I.S/")
            await page.wait_for_load_state('networkidle')

            # Wait for dashboard to load
            await page.wait_for_selector('input[placeholder*="boost productivity"]', timeout=10000)

            # Take initial screenshot
            await page.screenshot(path="trip_booking_initial_proper.png", full_page=True)
            print("✅ Initial UI screenshot captured")

            print("📝 Step 2: Entering command...")

            # Find and fill command input
            command_input = page.locator('input[placeholder*="boost productivity"]').first
            await command_input.click()
            await command_input.fill("book a 20 day trip to japan")

            # Take screenshot after command entry
            await page.screenshot(path="trip_booking_command_entered_proper.png", full_page=True)
            print("✅ Command entered screenshot captured")

            print("🚀 Step 3: Executing command...")

            # Find and click execute button
            execute_button = page.locator('button:has-text("Execute")').first
            await execute_button.click()

            print("⏳ Waiting for command execution results...")

            # Wait for command result to appear (the result card)
            # The result appears in a card with class containing command result text
            try:
                await page.wait_for_selector('text="Command Result"', timeout=300000)  # Wait up to 5 minutes
                print("✅ Command result card appeared!")

                # Take screenshot with results
                await page.screenshot(path="trip_booking_result_proper.png", full_page=True)
                print("✅ Post-execution result screenshot captured")

                # Wait a bit more for any additional updates
                await page.wait_for_timeout(2000)

                # Take final screenshot
                await page.screenshot(path="trip_booking_final_proper.png", full_page=True)
                print("✅ Final screenshot captured")

                print("🎯 Test completed successfully with RESULTS!")
                print("📁 Screenshots saved:")
                print("   - trip_booking_initial_proper.png")
                print("   - trip_booking_command_entered_proper.png")
                print("   - trip_booking_result_proper.png (WITH RESULTS)")
                print("   - trip_booking_final_proper.png")

                return True  # Success - results were shown

            except Exception as e:
                print(f"❌ Command result did not appear within timeout: {e}")
                await page.screenshot(path="trip_booking_no_results.png", full_page=True)
                print("📸 Screenshot taken of state without results")
                return False  # No results shown

        except Exception as e:
            print(f"❌ Test failed: {e}")
            await page.screenshot(path="test_error.png", full_page=True)
            return False

        finally:
            await browser.close()

def check_screenshot_for_results(screenshot_path):
    """Check if screenshot shows command results (placeholder - we'll use file existence and size as proxy)"""
    if not os.path.exists(screenshot_path):
        return False

    # For now, just check if the file exists and has reasonable size
    # In a real implementation, you'd use OCR or image analysis
    file_size = os.path.getsize(screenshot_path)
    print(f"📊 Screenshot {screenshot_path}: {file_size} bytes")

    # If file is too small, likely no content
    if file_size < 50000:  # Less than ~50KB probably empty/incomplete
        return False

    return True

async def run_test_loop():
    """Run test in a loop until results are properly shown"""
    max_attempts = 5
    attempt = 1

    while attempt <= max_attempts:
        print(f"\n🔄 ATTEMPT {attempt}/{max_attempts}")
        print("=" * 50)

        # Run the test
        success = await test_trip_booking_ui_with_results()

        if success:
            print("🎉 SUCCESS! Command results were displayed in UI!")
            return True

        print("❌ FAILURE: No command results shown in UI")

        # Check the screenshots that were created
        result_screenshot = "trip_booking_result_proper.png"
        if check_screenshot_for_results(result_screenshot):
            print("📸 Screenshots exist but results not detected - may need UI fixes")
        else:
            print("📸 Screenshots missing or incomplete")

        # Wait before retry
        print("⏳ Waiting 3 seconds before retry...")
        await asyncio.sleep(3)

        attempt += 1

    print(f"❌ FAILED after {max_attempts} attempts - manual intervention needed")
    return False

if __name__ == "__main__":
    asyncio.run(run_test_loop())
