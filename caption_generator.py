def format_text(text):
    # Split by underscore, capitalize each word, and join with spaces
    words = text.split('_')
    formatted = ' '.join(word.capitalize() for word in words)
    return formatted

# Example usage
input_text = "Price_profit_distribution"
output_text = format_text(input_text)
print(output_text)  # Output: Price Profit Distribution
