#!/bin/bash
# Helper script to generate MPI PBS scripts for different process counts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/run_mpi_template.sh"

# Create logs directory if it doesn't exist
mkdir -p "${SCRIPT_DIR}/../logs"

# Process counts to test: 2, 4, 8, 16, 32
NPROCS_LIST=(2 4 8 16 32)

for NPROCS in "${NPROCS_LIST[@]}"; do
    # Calculate nodes and ppn
    # For simplicity: use 1 node up to 32 cores (since we have 192 cores per node)
    if [ $NPROCS -le 32 ]; then
        NODES=1
        PPN=$NPROCS
    else
        # For larger runs, distribute across nodes
        NODES=$(( ($NPROCS + 31) / 32 ))
        PPN=32
    fi
    
    OUTPUT="${SCRIPT_DIR}/run_mpi_${NPROCS}.sh"
    
    echo "Generating MPI script for $NPROCS processes (nodes=$NODES, ppn=$PPN)..."
    
    # Replace placeholders in template
    sed -e "s/NODE_COUNT/$NODES/g" \
        -e "s/PPN_COUNT/$PPN/g" \
        -e "s/NPROCS/$NPROCS/g" \
        "$TEMPLATE" > "$OUTPUT"
    
    chmod +x "$OUTPUT"
    echo "  Created: $OUTPUT"
done

echo ""
echo "Generated MPI job scripts for process counts: ${NPROCS_LIST[*]}"
echo "Submit jobs with: qsub pbs_scripts/run_mpi_N.sh"
