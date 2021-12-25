
# curl http://localhost:5000/status/1# 
# {"id":1,  "status":"dispatched", "msg":"Order"}

# Dockerfile

.PHONY: test
test:
    pytest analytics/test/
run:
    python analytics/run.py --model=FD_DTree