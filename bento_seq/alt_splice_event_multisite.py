import numpy as np
import logging
from .read_distribution import ReadDistribution
from .bootstrap import gen_pdf_multisite
from . import BENTOSeqError

class AltSpliceEventMultisite(object):
    """ This class represents an alternative splice site event, with
    two or more alternative sites, defined
    by a name, the type of alternative splicing site, chromosome,
    strand, and the involved exons.

    **Parameters:**

    event_type : {'A5SS', 'A3SS'}

    event_id : string

    chromosome : string
        The chromosome where the event is located. Must match the
        chromosome names in the BAM-file.

    strand : {'+', '-'}

    const_exon : tuple
        A tuple specifying the constitutive exon in the format ``(start, end)``.

    exons : array_like
        A list/tuple of alternative exons in the format
        ``[(exon_1_start, exon_1_end), (exon_2_start, exon_2_end), ... ]``.
        The number of exons must be two or more. By default,
        intervals are assumed to be in Python format, *i.e.* 0-based
        and right-open. Use the ``one_based_pos`` option if your
        coordinates are 1-based and closed.

    one_based_pos : bool (default=False)
        Whether to interpret the exon coordinates as 1-based and
        closed intervals. This is often the default format for genomic
        coordinates, *e.g.* on the UCSC Genome Browser. By default,
        coordiantes are assumed to be 0-based and right-open. If this
        option is True, then exon coordinates are converted to 0-based
        indexing on initialization. All internal calculations are
        based on 0-based coordinates.
    
    """

    def __init__(self, event_type, event_id, chromosome, strand, const_exon, alt_exons, one_based_pos=False):
        event_type = event_type.upper()

        if event_type not in ('A5SS', 'A3SS'):
            raise BENTOSeqError("Unknown alternative splicing event type: %s" % str(event_type))

        if len(alt_exons) < 2:
            raise BENTOSeqError("Need at least two alternative exons.")

        for i, exon in enumerate(alt_exons):
            if len(exon) != 2:
                raise BENTOSeqError("Exon #%d has wrong length: %d (must be 2)." % len(exon))

        if strand not in ('-', '+'):
            raise BENTOSeqError("Unknown strand type: %s (must be '+' or '-')." % str(strand))

        self.event_type = event_type
        self.event_id = event_id
        self.chromosome = chromosome
        self.strand = strand
        self.const_exon = const_exon
        self.alt_exons = alt_exons

        if one_based_pos:
            self.const_exon = (self.const_exon[0] - 1, self.const_exon[1])
            self.alt_exons = [(e[0] - 1, e[1]) for e in self.alt_exons]

        if strand == '-':
            self.const_exon = (-self.const_exon[1] + 1, -self.const_exon[0] + 1)
            self.alt_exons = [(-e[1] + 1, -e[0] + 1) for e in self.alt_exons]

        for a1, a2 in zip(self.alt_exons[::2], self.alt_exons[1::2]):
            if not (a2[0] > a1[0] or a2[1] > a1[1]):
                raise BENTOSeqError("Exons must be listed from 5' to 3' "
                                    "on the transcribed strand.")

        if strand == '+':
            if self.event_type == 'A3SS':
                self.junctions = [(self.chromosome, self.const_exon[1], ae[0])
                                  for ae in self.alt_exons]
            elif self.event_type == 'A5SS':
                self.junctions = [(self.chromosome, ae[1], self.const_exon[0])
                                  for ae in self.alt_exons]
            else:
                raise BENTOSeqError
        elif strand == '-':
            if self.event_type == 'A3SS':
                self.junctions = [(self.chromosome, -ae[0] + 1, -self.const_exon[1] + 1)
                                  for ae in self.alt_exons]
            elif self.event_type == 'A5SS':
                self.junctions = [(self.chromosome, self.const_exon[0] + 1, -ae[1] + 1)
                                  for ae in self.alt_exons]
        else:
            raise BENTOSeqError

        self.const_exon_length = self.const_exon[1] - self.const_exon[0]
        self.alt_exons_lengths = [e[1] - e[0] for e in self.alt_exons]

        if self.event_type == 'A5SS':
            if not self.const_exon[0] > self.alt_exons[-1][1]:
                raise BENTOSeqError("Event is not a valid A5SS event.")

            for a1, a2 in zip(self.alt_exons[::2], self.alt_exons[1::2]):
                if not (a1[0] == a2[0] and
                        a2[0] > a1[1]):
                    raise BENTOSeqError("Event is not a valid A5SS event.")
        elif self.event_type == 'A3SS':
            if not self.const_exon[1] < self.alt_exons[0][0]:
                raise BENTOSeqError("Event is not a valid A3SS event.")
            for a1, a2 in zip(self.alt_exons[::2], self.alt_exons[1::2]):
                if not (a1[1] == a2[1] and
                        a1[0] < a2[0]):
                    raise BENTOSeqError("Event is not a valid A3SS event.")
        else:
            raise BENTOSeqError

    def build_read_distribution(self, bamfile, min_overhang=5,
                                max_edit_distance=2,
                                max_num_mapped_loci=1):

        """Build the read distribution for this event from a BAM-file.

        **Parameters:**

        bamfile : :py:class:`pysam.Samfile`
            Reference to a binary, sorted, and indexed SAM-file.

        min_overhang : int (default=5)
            Minimum overhang on either side of the splice junction
            required for counting a read towards the read distribution.

        max_edit_distance : int (default=2)
            Maximum edit distance (number of mismatches against the
            reference genome) to allow before skipping a read.

        max_num_mapped_loci : int (default=1)
            Indiciates to how many locations a read may be aligned to
            be a counted. By default, only uniquely mappable reads are
            alowed.

        """

        self.junction_read_distributions = []
        for junction in self.junctions:
            read_distribution = \
                ReadDistribution.from_junction(
                    bamfile, junction,
                    max_edit_distance,
                    max_num_mapped_loci)

            if read_distribution.is_empty:
                logging.debug("Event %s: No reads in %s "
                              "map to junction %s:%d:%d." %
                              (self.event_id, bamfile.filename,
                               junction[0], junction[1], junction[2]))

            positions, reads = zip(
                *read_distribution.to_list(min_overhang))
            if self.strand == '-': reads = reads[::-1]
            self.junction_read_distributions.append(reads)

    def bootstrap_event(self, n_bootstrap_samples=1000, n_grid_points=100,
                        a=1, b=1, r=0):

        """Estimate PSI (percent spliced-in) value for this event.

        **Parameters:**

        n_bootstrap_samples : int (default=1000)
            How many bootstrap samples to use for estimation of PSI.

        n_grid_points : 100 (default=100)
            How many points to use for the numerical approximation of
            the bootstrap probability density function.

        a : int (default=1)
            Bayesian pseudo-count for the inclusion reads.

        b : int (default=1)
            Bayesian pseudo-count for the exclusion reads.

        r : int (default=0)
            Bayesian pseudo-count for the normalization of the
            bootstrap probability density function.

        **Returns:**

        n_inc : int
            The number of reads mapped to the inclusion junctions.

        n_exc : int
            The number of reads mapped to the exclusion junctions.

        p_inc : int
            Number of inclusion mapping positions.

        p_exc : int
            Number of exclusion mapping positions.

        psi_standard : float
            Naive PSI estimate.

        psi_bootstrap : float
            Bootstrap estimate of PSI.

        psi_bootstrap_std : float
            Estimated standard deviation of ``psi_bootstrap``.
        """
    
        reads = [np.array(x) for x in self.junction_read_distributions]

        n_reads = np.array([x.sum() for x in reads])

        p_reads = np.array([x.size for x in reads])

        min_p = min(*p_reads)
        scaled_n_reads = n_reads.astype(np.float64) / p_reads * min_p

        psi_standard = (scaled_n_reads + 1) / (scaled_n_reads.sum() + scaled_n_reads.size)

        psi_bootstrap, psi_std = gen_pdf_multisite(reads, n_bootstrap_samples, n_grid_points, a, b, r)

        return n_reads, p_reads, psi_standard, psi_bootstrap, psi_std
