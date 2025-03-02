def rabin_karp(pattern, text, d=256, q=101):
    """
    Implementation of the Rabin-Karp string matching algorithm.
    
    Args:
        pattern (str): Pattern to search for
        text (str): Text to search in
        d (int): Number of characters in the alphabet
        q (int): A prime number for hash calculation
        
    Returns:
        list: List of starting indices where pattern was found
    """
    m = len(pattern)
    n = len(text)
    p = 0  # hash value for pattern
    t = 0  # hash value for text
    h = 1
    matches = []
    
    # Calculate h = d^(m-1) % q
    for i in range(m-1):
        h = (h * d) % q
    
    # Calculate initial hash values
    for i in range(m):
        p = (d * p + ord(pattern[i])) % q
        t = (d * t + ord(text[i])) % q
    
    # Slide pattern over text one by one
    for i in range(n - m + 1):
        # Check if hash values match
        if p == t:
            # Check characters one by one
            match = True
            for j in range(m):
                if text[i + j] != pattern[j]:
                    match = False
                    break
            
            if match:
                matches.append(i)
        
        # Calculate hash value for next window
        if i < n - m:
            t = (d * (t - ord(text[i]) * h) + ord(text[i + m])) % q
            
            # Make sure hash value is positive
            if t < 0:
                t = t + q
    
    return matches 