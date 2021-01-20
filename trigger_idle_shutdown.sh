#/usr/bin/env bash
# Idle shutdown trigger script
# Script to automatically shutdown a Google Cloud Compute Node that has more than three minutes of idle time

# compute load threshold
threshold=0.1
enableSlackHook=0

while getopts ":s" opt; do
case ${opt} in
s)
        enableSlackHook=1
;;
*)
        enableSlackHook=0
esac
done

## Check the instance name and ignore for api-dev
instance_name=$(curl "http://metadata.google.internal/computeMetadata/v1/instance/name?alt=text" -H "Metadata-Flavor: Google")
if [[ ${instance_name} = "api" ]]
then
    echo "Cannot trigger shutdown of api-dev, so exiting"
    exit
fi

## Setup the Slack hook
## Send the message only if enable_slack file is present
SLACK_HOOK='https://hooks.slack.com/services/H9KIW1Y5U/BDFGGHT9/KUITOWP234fM56ZMRUAB56U3E'
slackit() {
    TYPE_OF_MESG="$1"
    MESG="$2"
    if [ ${enableSlackHook} -eq 1 ]
    then
        curl -X POST -H 'Content-type: application/json' --data "{ \"text\":\"${TYPE_OF_MESG}:${MESG}\n\" }" ${SLACK_HOOK}
    else
	echo "\{${TYPE_OF_MESG}\}:${MESG}"
    fi
}

## Logic to check for inactive count
inactivecount=0
while true
do
  load=$(uptime | sed -e 's/.*load average: //g' | awk '{ print $3 }')
  res=$(awk 'BEGIN{ print "'$load'"<"'$threshold'" }')
  if (( $res ))
  then
    echo "Idling.."
    ((inactivecount+=1))
  else
      slackit "Info" "Cancelling countdown as $USER has activity on $(hostname)"
      ((inactivecount=0))
  fi
  echo "Idle minutes count = $inactivecount"
  if (( inactivecount==5 ))
  then
	  echo Shutting down $inactivecount
	  slackit "Warning" "Machine $USER @ $(hostname) shutting down if machine remains idle in next 3 minutes" #Actually the machine will shutdown in 5 minutes
  fi
  if (( inactivecount>9 ))
  then
    echo Shutting down
    # wait a little bit more before actually pulling the plug
    slackit "Final" "Shutting down $USER @ $(hostname) now"
    sleep 60
    sudo poweroff
    exit
  fi
  sleep 60
done

