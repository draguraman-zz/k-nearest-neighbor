use strict;

my $goldfile = shift @ARGV;
my $ansfile = shift @ARGV;

my %doc_tag = ();

open(GOLDFILE, "<$goldfile");

while(<GOLDFILE>)
{
    chomp;
    my @arr = split;
    $doc_tag{$arr[1]} = $arr[0];
}

close(GOLDFILE);

my $total = 0;
my $correct = 0;

open(ANSFILE, "<$ansfile");

while(<ANSFILE>)
{
    chomp;
    my @arr = split;
    if($#arr == 1)
    {
	$total++;
	if($arr[0] == $doc_tag{$arr[1]})
	{
	    $correct++;
	}
    }
}

close(ANSFILE);

print "accuracy: ", $correct/$total, "\n";
