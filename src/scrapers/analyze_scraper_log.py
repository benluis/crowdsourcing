import re
import sys
import os
import glob
from collections import defaultdict
from datetime import datetime
import statistics

def analyze_scraper_output(log_file_path, output_dir="data/scraped"):
    print(f"=== SCRAPER RUN ANALYSIS: {log_file_path} ===\n")
    
    if not os.path.exists(log_file_path):
        print(f"Error: Log file '{log_file_path}' not found.")
        return

    # --- 1. PARSE LOG INTO EVENT STREAM ---
    # We'll read the file once and build a structured timeline of events
    events = []
    
    # Regex Patterns
    p_timestamp = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    p_start = re.compile(r"\[PROJECT_START\] Processing project (\d+)")
    p_end = re.compile(r"\[PROJECT_END\] Finished .* Status=(\w+)")
    # Matches: "Sleeping for 5s", "Sleeping 10s", "Pausing for 60 seconds"
    p_sleep = re.compile(r"(?:Sleeping|Pausing) (?:for )?(\d+)(?:s| seconds)")
    p_reset = re.compile(r"Resetting (?:scraper )?session")
    p_metric = re.compile(r"\[METRIC\] .* Fetched (\d+) (comments|updates)")
    p_error = re.compile(r"\[ERROR\] (.*)")
    
    start_time = None
    last_time = None
    
    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Extract timestamp
            m_ts = p_timestamp.search(line)
            ts = None
            if m_ts:
                try:
                    ts = datetime.strptime(m_ts.group(1), "%Y-%m-%d %H:%M:%S")
                    if start_time is None: start_time = ts
                    last_time = ts
                except:
                    pass
            
            # Identify Event Type
            if p_start.search(line):
                m = p_start.search(line)
                events.append({'type': 'START', 'id': m.group(1), 'ts': ts})
            elif p_end.search(line):
                m = p_end.search(line)
                events.append({'type': 'END', 'status': m.group(1), 'ts': ts})
            elif p_sleep.search(line):
                m = p_sleep.search(line)
                duration = int(m.group(1))
                events.append({'type': 'SLEEP', 'duration': duration, 'ts': ts})
            elif p_reset.search(line):
                events.append({'type': 'RESET', 'ts': ts})
            elif p_metric.search(line):
                m = p_metric.search(line)
                events.append({'type': 'METRIC', 'count': int(m.group(1)), 'kind': m.group(2), 'ts': ts})
            elif p_error.search(line):
                m = p_error.search(line)
                events.append({'type': 'ERROR', 'msg': m.group(1), 'ts': ts})

    if not events:
        print("No events found in log.")
        return

    # --- 2. RECONSTRUCT PROJECT STATES ---
    # We want to know for each project: Was it clean? Did it have retries? Did it fail?
    projects = []
    current_project = None
    
    # Global stats
    total_sleep_time = 0
    sleep_counts = defaultdict(int)
    
    for e in events:
        if e['type'] == 'START':
            # If we were already in a project (maybe it crashed or didn't log END), close it
            if current_project:
                projects.append(current_project)
            
            current_project = {
                'id': e['id'],
                'start_ts': e['ts'],
                'end_ts': None,
                'status': 'Unknown',
                'sleeps': [],
                'errors': [],
                'metrics': {'comments': 0, 'updates': 0},
                'reset_during': False
            }
        
        elif e['type'] == 'END':
            if current_project:
                current_project['status'] = e['status']
                current_project['end_ts'] = e['ts']
                projects.append(current_project)
                current_project = None
        
        elif e['type'] == 'SLEEP':
            total_sleep_time += e['duration']
            sleep_counts[e['duration']] += 1
            if current_project:
                current_project['sleeps'].append(e['duration'])
            else:
                # Sleep happened between projects (e.g. the 60s safety pause)
                # We can record this as a "System Event" or attach to previous/next
                pass
                
        elif e['type'] == 'RESET':
            if current_project:
                current_project['reset_during'] = True
                
        elif e['type'] == 'METRIC':
            if current_project:
                current_project['metrics'][e['kind']] += e['count']
                
        elif e['type'] == 'ERROR':
            if current_project:
                current_project['errors'].append(e['msg'])

    # Close last project if open
    if current_project:
        projects.append(current_project)

    # --- 3. ANALYZE RHYTHM & TIPPING POINTS ---
    
    # Classify each project run
    # - Clean: Success, No Sleeps, No Errors
    # - Soft Blocked: Success, but had sleeps < 60s
    # - Hard Blocked: Had sleeps >= 60s (or multiple small ones totaling high)
    # - Failed: Status != Success
    
    run_sequence = []
    
    for p in projects:
        max_sleep = max(p['sleeps']) if p['sleeps'] else 0
        total_sleep = sum(p['sleeps'])
        
        if p['status'] != 'Success':
            state = 'FAILED'
        elif total_sleep == 0:
            state = 'CLEAN'
        elif max_sleep >= 60 or total_sleep >= 60:
            state = 'HARD_BLOCK'
        else:
            state = 'SOFT_BLOCK' # e.g. 5s, 10s delays
            
        p['state'] = state
        run_sequence.append(state)

    # --- 4. GENERATE REPORT ---
    
    # Basic Stats
    print(f"Total Projects: {len(projects)}")
    print(f"Total Runtime:  {(last_time - start_time).total_seconds() / 3600:.2f} hours")
    print(f"Total Sleep:    {total_sleep_time} seconds ({total_sleep_time/60:.1f} minutes)")
    
    # Sleep Distribution
    print("\n=== SLEEP DISTRIBUTION ===")
    for duration, count in sorted(sleep_counts.items()):
        print(f"  {duration}s: {count} times")

    # Tipping Point Analysis
    # Find the longest streak of CLEAN projects at the start
    first_non_clean = next((i for i, state in enumerate(run_sequence) if state != 'CLEAN'), None)
    
    print("\n=== TIPPING POINT ANALYSIS ===")
    if first_non_clean is None:
        print("  Status: PERFECT RUN (No rate limits detected)")
    else:
        print(f"  First Rate Limit at Project #{first_non_clean + 1}")
        print(f"  Clean Streak: {first_non_clean} projects")
        
        # Calculate speed during the clean streak
        if first_non_clean > 1:
            clean_p = projects[:first_non_clean]
            if clean_p[-1]['end_ts'] and clean_p[0]['start_ts']:
                duration = (clean_p[-1]['end_ts'] - clean_p[0]['start_ts']).total_seconds()
                rpm = first_non_clean / (duration / 60)
                print(f"  Clean Speed:  {rpm:.2f} projects/minute ({(duration/first_non_clean):.2f}s per project)")
                
                # Recommendation
                print(f"  --> RECOMMENDATION: Set sleep to maintain ~{rpm * 0.8:.2f} RPM")
                target_sec = (60 / (rpm * 0.8))
                print(f"      Target seconds/project: {target_sec:.2f}s")
                # Estimate current overhead (network time)
                avg_net_time = duration / first_non_clean
                suggested_sleep = max(0.5, target_sec - avg_net_time + 1.0) # Add 1s buffer
                print(f"      Suggested added sleep:  {suggested_sleep:.2f}s")

    # Visual Run Chart (Compressed)
    print("\n=== STATE RUN CHART ===")
    print("Legend: [CLEAN] = No delays | [SOFT] = Small delays (5-30s) | [HARD] = Long delays (>60s) | [FAIL] = Error")
    
    compressed_chart = []
    current_state = None
    count = 0
    
    for state in run_sequence:
        if state != current_state:
            if current_state:
                compressed_chart.append(f"[{count} {current_state}]")
            current_state = state
            count = 1
        else:
            count += 1
    if current_state:
        compressed_chart.append(f"[{count} {current_state}]")
        
    # Print wrapped
    line_buffer = ""
    for item in compressed_chart:
        if len(line_buffer) + len(item) + 4 > 100:
            print(line_buffer)
            line_buffer = ""
        line_buffer += item + " -> "
    print(line_buffer)

    # Session Reset Efficacy
    print("\n=== SESSION RESET EFFICACY ===")
    resets = [i for i, p in enumerate(projects) if p['reset_during']]
    if not resets:
        print("No session resets detected.")
    else:
        print(f"Total Resets: {len(resets)}")
        success_resets = 0
        for idx in resets:
            # Check the next 5 projects
            upcoming = run_sequence[idx+1 : idx+6]
            if not upcoming: continue
            
            # If the next few are CLEAN or SOFT (not HARD/FAIL), it worked
            if all(s in ['CLEAN', 'SOFT_BLOCK'] for s in upcoming):
                success_resets += 1
                
        print(f"Resets followed by stable runs: {success_resets}/{len(resets)} ({success_resets/len(resets)*100:.1f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_scraper_log.py <path_to_log_file>")
    else:
        analyze_scraper_output(sys.argv[1])
