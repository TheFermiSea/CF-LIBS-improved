with open('cflibs/inversion/boltzmann.py', 'r') as f:
    lines = f.readlines()

# Find and fix corrupted lines
for i in range(len(lines)):
    if '[...417 lines truncated' in lines[i]:
        # Remove this line
        lines[i] = ''

# Join and write back
content = ''.join(lines)
# Fix the y_pred line that has garbage prepended
content = content.replace('[...417 lines truncated. DO NOT paste truncated content into edit_file. Use read_file with start_line/end_line.]            y_pred = m * x + c', '            y_pred = m * x + c')

with open('cflibs/inversion/boltzmann.py', 'w') as f:
    f.write(content)
print('Fixed!')
