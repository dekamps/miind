#
# Regular cron jobs for the miind package
#
0 4	* * *	root	[ -x /usr/bin/miind_maintenance ] && /usr/bin/miind_maintenance
